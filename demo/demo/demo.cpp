#include <iostream>
#include <csignal>
#include <stdlib.h>     /* srand, rand */
#include <mutex>
#include <data.h>
#include <string>
#include <camera_elaboration.h>
#include <configuration.h>
#include "zmq.hpp"
#include <sys/stat.h>
#include <fcntl.h>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;
int NUM_ITERS = 400;

zmq::socket_t *app_socket;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
    if (app_socket != nullptr) {
        app_socket->close();
    }
    FatalError("Closing application");
}

edge::camera prepareCamera(int camera_id, std::string &net, char &type, int &n_classes, bool show) {
    YAML::Node config = YAML::LoadFile("../../data/all_cameras_en.yaml");
    YAML::Node cameras_yaml = config["cameras"];

    net = config["net"].as<std::string>();
    type = config["type"].as<char>();
    n_classes = config["classes"].as<int>();
    std::string tif_map_path = config["tif"].as<std::string>();
    std::string password = "";
    if(config["password"])
        password = config["password"].as<std::string>();

    edge::camera_params camera_par;
    for (auto && cam_yaml : cameras_yaml) {
        int ref_cam_id = cam_yaml["id"].as<int>();
        if (ref_cam_id != camera_id) continue;
        camera_par.id = ref_cam_id;
        if (cam_yaml["encrypted"].as<int>()) {
	    if(password == "") {
                std::cout<<"Please insert the password to decrypt the cameras input"<<std::endl;
                std::cin>>password;
            }
            camera_par.input = decryptString(cam_yaml["input"].as<std::string>(), password);
            //camera_par.input = decryptString(cam_yaml["input"].as<std::string>(),);
            /*std::cout << "The input file is encrypted. Throwing exception" << std::endl;
            throw;*/
        } else {
            camera_par.input = cam_yaml["input"].as<std::string>();
        }
        camera_par.pmatrixPath        = cam_yaml["pmatrix"].as<std::string>();
        camera_par.maskfilePath       = cam_yaml["maskfile"].as<std::string>();
        camera_par.cameraCalibPath    = cam_yaml["cameraCalib"].as<std::string>();
        camera_par.maskFileOrientPath = cam_yaml["maskFileOrient"].as<std::string>();
        camera_par.show               = show;
        if (cam_yaml["tif"]) {
            tif_map_path = cam_yaml["tif"].as<std::string>();
        }
        break;
    }

    if (!camera_par.id) {
        std::cout << "No camera data could be found with given id " << camera_id << std::endl;
        throw;
    }

    std::cout << "Camera parameters read!" << std::endl << camera_par << std::endl;
    std::cout << "Using TIF at " << tif_map_path << std::endl;

    edge::Dataset_t dataset;
    switch (n_classes) {
        case 10: dataset = edge::Dataset_t::BDD; break;
        case 80: dataset = edge::Dataset_t::COCO; break;
        default: FatalError("Dataset type not supported yet, check number of classes in parameter file.");
    }

    edge::camera camera;
    std::cout << "Reading calibration matrix in " << camera_par.cameraCalibPath << std::endl;
    readCalibrationMatrix(camera_par.cameraCalibPath, camera.calibMat, camera.distCoeff, camera.calibWidth,
                          camera.calibHeight);
    std::cout << "Calibration matrix read!" << std::endl << camera.calibMat << std::endl;
    std::cout << "Reading projection matrix in " << camera_par.pmatrixPath << std::endl;
    readProjectionMatrix(camera_par.pmatrixPath, camera.prjMat);
    std::cout << "Projection matrix read!" << std::endl << camera.prjMat << std::endl;
    camera.id = camera_par.id;
    camera.input = camera_par.input;
    camera.streamWidth = config["width"].as<int>();
    camera.streamHeight = config["height"].as<int>();
    camera.show = true; // THIS SHOULD BE INPUT
    camera.invPrjMat = camera.prjMat.inv();
    camera.dataset = dataset;
    std::cout << "Inverse Projection matrix!" << std::endl << camera.invPrjMat << std::endl;

    camera.adfGeoTransform = (double *) malloc(6 * sizeof(double));
    readTiff(tif_map_path, camera.adfGeoTransform);
    std::cout << "Using following point as reference in initialiseReference in geoConv (lat: "
    << camera.adfGeoTransform[3] << ", lon: " << camera.adfGeoTransform[0] << ")" << std::endl;
    camera.geoConv.initialiseReference(camera.adfGeoTransform[3], camera.adfGeoTransform[0], 0);
    return camera;
}

char* prepareMessage(std::vector<tk::dnn::box> &box_vector, std::vector<std::tuple<double, double>> &coords,
                     // std::vector<std::tuple<double, double>> &coordsGeo,
                     std::vector<std::tuple<double, double, double, double, double, double, double, double>> &boxCoords,
                     unsigned int frameAmount, int cam_id, double lat_init, double lon_init, unsigned int *size) {
    /*box_vector.erase(std::remove_if(box_vector.begin(), box_vector.end(), [](tk::dnn::box &box) {
        return box.cl == 7 || box.cl == 8;
    }), box_vector.end()); // if traffic signs or traffic lights*/
    for (int i = box_vector.size() - 1; i >= 0; i--) {
        // if traffic signs or traffic lights
        /*std::cout << "In removing boxes: pixel x: " << box_vector[i].x << " pixel y: " << box_vector[i].y <<
            " north: " << std::get<0>(coords[i]) << " east: " << std::get<1>(coords[i]) <<
            " lat: " << std::get<0>(coordsGeo[i]) << " lon: " << std::get<1>(coordsGeo[i]) << std::endl;*/
        if (box_vector[i].cl == 7 || box_vector[i].cl == 8) {
            box_vector.erase(box_vector.begin()+i);
            coords.erase(coords.begin()+i);
            //coordsGeo.erase(coordsGeo.begin()+i);
            boxCoords.erase(boxCoords.begin()+i);
        }
    }
    *size = box_vector.size() * (sizeof(double) * 10 + sizeof(int) + 1 + sizeof(float) * 4) + 1 + sizeof(int)
            + sizeof(unsigned long long) + sizeof(double) * 2;
    char *data = (char *) malloc(*size);
    char *data_origin = data;
    char flag = ~0;
    memcpy(data++, &flag, 1);
    memcpy(data, &cam_id, sizeof(int));
    data += sizeof(int);
    unsigned long long timestamp = getTimeMs();
    memcpy(data, &timestamp, sizeof(unsigned long long));
    data += sizeof(unsigned long long);
    memcpy(data, &lat_init, sizeof(double));
    data += sizeof(double);
    memcpy(data, &lon_init, sizeof(double));
    data += sizeof(double);
    for (int i = 0; i < box_vector.size(); i++) {
        tk::dnn::box box = box_vector[i];
        std::tuple<double, double> coord = coords[i];
        double north = std::get<0>(coord);
        double east = std::get<1>(coord);
        /*std::tuple<double, double> coordGeo = coordsGeo[i];
        double lat = std::get<0>(coordGeo);
        double lon = std::get<1>(coordGeo);*/
        memcpy(data, &north, sizeof(double));
        data += sizeof(double);
        memcpy(data, &east, sizeof(double));
        data += sizeof(double);
        /*memcpy(data, &lat, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lon, sizeof(double));
        data += sizeof(double);*/
        memcpy(data, &frameAmount, sizeof(unsigned int));
        data += sizeof(unsigned int);
        memcpy(data, &box.cl, sizeof(char));
        data += sizeof(char);
        memcpy(data, &box.x, sizeof(float));
        data += sizeof(float);
        memcpy(data, &box.y, sizeof(float));
        data += sizeof(float);
        memcpy(data, &box.w, sizeof(float));
        data += sizeof(float);
        memcpy(data, &box.h, sizeof(float));
        data += sizeof(float);
        std::tuple<double, double, double, double, double, double, double, double> boxCoord = boxCoords[i];
        double lat_ur = std::get<0>(boxCoord);
        double lon_ur = std::get<1>(boxCoord);
        double lat_lr = std::get<2>(boxCoord);
        double lon_lr = std::get<3>(boxCoord);
        double lat_ll = std::get<4>(boxCoord);
        double lon_ll = std::get<5>(boxCoord);
        double lat_ul = std::get<6>(boxCoord);
        double lon_ul = std::get<7>(boxCoord);
        memcpy(data, &lat_ur, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lon_ur, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lat_lr, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lon_lr, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lat_ll, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lon_ll, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lat_ul, sizeof(double));
        data += sizeof(double);
        memcpy(data, &lon_ul, sizeof(double));
        data += sizeof(double);
    }
    // printf("\n");
    return data_origin;
}

int main(int argc, char *argv[]) {
    std::cout<<"detection!\n";
    signal(SIGINT, sig_handler);

    bool use_socket = true;
    // bool use_pipe = true;
    if (argc > 1) {
        // use_pipe = atoi(argv[1]);
        use_socket = atoi(argv[1]);
    }
    int camera_id = 20939;
    if (argc > 2) {
        camera_id = atoi(argv[2]);
    }
    std::string socketPort = "5559";
    int argv_ref = 2;
    if (use_socket) {
        if (argc > ++argv_ref) {
            socketPort = argv[argv_ref];
            // pipePathWrite = argv[argv_ref];
        }
        // if (argc > ++argv_ref) {
        //     pipePathRead = argv[argv_ref];
        // }
    }
    /*
    std::string net = "yolo3_berkeley_fp32.rt";
    if(argc > 3)
        net = argv[2];
    char ntype = 'y';
    if(argc > 4)
        ntype = argv[4][0];
    int n_classes = 80;
    if(argc > 5)
        n_classes = atoi(argv[5]);
    int n_batch = 1;
    if(argc > 6)
        n_batch = atoi(argv[6]);
    */
    bool show = false;
    if(argc > ++argv_ref)
        show = atoi(argv[argv_ref]);

    if (argc > ++argv_ref) {
        SAVE_RESULT = atoi(argv[argv_ref]);
    }

    int n_batch = 1;
    std::string net;
    char ntype;
    int n_classes;

    edge::camera camera = prepareCamera(camera_id, net, ntype, n_classes, show);

    if (n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    //if (!show)
    //    SAVE_RESULT = true;

    double north, east;

    tk::dnn::Yolo3Detection yolo;
    tk::dnn::CenternetDetection cnet;
    tk::dnn::MobilenetDetection mbnet;  

    tk::dnn::DetectionNN *detNN;  

    switch(ntype)
    {
        case 'y':
            detNN = &yolo;
            break;
        case 'c':
            detNN = &cnet;
            break;
        case 'm':
            detNN = &mbnet;
            n_classes++;
            break;
        default:
            FatalError("Network type not allowed (3rd parameter)\n");
    }

    detNN->init(net, n_classes, n_batch);

    gRun = true;

    std::cout << "Opening VideoCapture for input " << camera.input << std::endl;
    cv::VideoCapture cap(camera.input);
    if(!cap.isOpened()) {
        std::cout << "Camera could not be started." << std::endl;
        exit(1);
    } else {
        std::cout << "Camera started" << std::endl;
    }

    cv::VideoWriter resultVideo;
    if(SAVE_RESULT) {
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        resultVideo.open("result.mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }

    cv::Mat frame;
    if (show) {
        std::cout << "Opening window..." << std::endl;
        cv::namedWindow("detection", cv::WINDOW_NORMAL);
        cv::resizeWindow("detection", 1024, 800);
        std::cout << "Window successfully opened" << std::endl;
    }

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    zmq::message_t unimportant_message;
    zmq::context_t context(1);
    if (use_socket) {
        app_socket = new zmq::socket_t(context, ZMQ_REQ);
        std::cout << "Connecting to tcp://0.0.0.0:" << socketPort << std::endl;
        app_socket->bind("tcp://0.0.0.0:" + socketPort);

        /* Check if named pipes exist and create them */
        // unlink(pipePathWrite);
        // unlink(pipePathRead);
        // if (mkfifo(pipePathWrite, 0666) < 0) {
        //     perror("Error creating write pipe");
        // }
        // if (mkfifo(pipePathRead, 0666) < 0) {
        //     perror("Error creating read pipe");
        // }

        /*std::cout << "Opening FIFO WRITE PIPE in " << pipePathWrite << std::endl;
        if ((fifo_write = open(pipePathWrite, O_WRONLY)) < 0) {
            perror("Error opening write pipe");
            FatalError("Pipe for write not opened");
        }
        std::cout << "Opening FIFO READ PIPE in " << pipePathRead << std::endl;
        if ((fifo_read = open(pipePathRead, O_RDONLY)) < 0) {
            perror("Error opening read pipe");
            FatalError("Pipe for read not opened");
        }*/
    } else {
        std::cout << "Not opening socket" << std::endl;
        // std::cout << "Not opening pipes" << std::endl;
    }

    unsigned int frameAmount = 0;
    float scale_x   = camera.hasCalib ? camera.calibWidth  / camera.streamWidth : 1;
    float scale_y   = camera.hasCalib ? camera.calibHeight / camera.streamHeight: 1;
    std::vector<tk::dnn::box> box_vector;
    std::vector<std::tuple<double, double>> coords;
    std::vector<std::tuple<double, double>> coordsGeo;
    std::vector<std::tuple<double, double, double, double, double, double, double, double>> boxCoords;
    double lat_ur, lat_lr, lat_ll, lat_ul, lon_ur, lon_lr, lon_ll, lon_ul;
    double lat, lon;
    int iters = 0;
    while(iters < NUM_ITERS) {
        batch_dnn_input.clear();
        batch_frame.clear();
        
        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame; 
            if(!frame.data)
                continue;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        }
        if(!frame.data) 
            continue;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        for (auto &box_batch : detNN->batchDetected) {
            for (auto &box : box_batch) {
                convertCameraPixelsToMapMeters((box.x + box.w / 2)*scale_x, (box.y + box.h)*scale_y, box.cl, camera, north, east);
                //convertCameraPixelsToMapMeters(box.x + box.w/2, box.y + box.h/2, box.cl, camera, north,
                //                               east); // box center
                // convertCameraPixelsToGeodetic(box.x + box.w/2, box.y + box.h/2, box.cl, camera, lat,
                //                              lon); // box center
                convertCameraPixelsToGeodetic(box.x + box.w, box.y, box.cl, camera, lat_ur,
                                              lon_ur); // box upper right corner
                convertCameraPixelsToGeodetic(box.x + box.w, box.y + box.h, box.cl, camera, lat_lr,
                                              lon_lr); // box lower right corner
                convertCameraPixelsToGeodetic(box.x, box.y + box.h, box.cl, camera, lat_ll,
                                              lon_ll); // box lower left corner
                convertCameraPixelsToGeodetic(box.x, box.y, box.cl, camera, lat_ul,
                                              lon_ul); // box upper left corner
                box_vector.push_back(box);
                coords.push_back(std::make_tuple(north, east));
                // coordsGeo.push_back(std::make_tuple(lat, lon));
                boxCoords.push_back(std::make_tuple(lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul));
                /*std::cout << "box.x " << box.x << "box.y " << box.y << " lat_ur " << lat_ur << " lon_ur " << lon_ur
                    << " lat_lr " << lat_lr << " lon_lr " << lon_lr << " lat_ll " << lat_ll << " lon_ll " << lon_ll
                    << " lat_ul " << lat_ul << " lon_ul " << lon_ul << std::endl;*/
                // printf("\t(%f,%f) converted to (%f,%f)\n", box.x, box.y, north, east);
            }
        }
        detNN->draw(batch_frame);

        if (show) {
            for (int bi=0; bi < n_batch; ++bi) {
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }

        // send thru socket or pipe
        if (use_socket) {
            unsigned int size;
            // char *data = prepareMessage(box_vector, coords, coordsGeo, boxCoords, frameAmount, camera_id, &size);
            char *data = prepareMessage(box_vector, coords, boxCoords, frameAmount, camera_id, 
					camera.adfGeoTransform[3], camera.adfGeoTransform[0], &size);
            zmq::message_t message(size);
            memcpy(message.data(), data, size);
            std::cout << "[" << frameAmount << "] Waiting for message..." << std::endl;
            app_socket->send(message, 0);
            free(data);
            app_socket->recv(&unimportant_message); // wait for python workflow to ack

            // std::cout << "[" << frameAmount << "] Waiting for message..." << std::endl;
            // if ((fifo_write = open(pipePathWrite, O_WRONLY)) < 0) {
            //     perror("Error opening write pipe");
            //     FatalError("Pipe for write not opened");
            // }
            // if ((fifo_read = open(pipePathRead, O_RDONLY)) < 0) {
            //     perror("Error opening read pipe");
            //     FatalError("Pipe for read not opened");
            // }
            // if (write(fifo_write, data, size) < 0) {
            //     perror("Error when writing");
            // }
            // // std::cout << "AFTER WRITING from FIFO PIPE" << std::endl;
            // close(fifo_write);
            // free(data);

            // char buff[1];
            // // std::cout << "READING from FIFO PIPE" << std::endl;
            // if (read(fifo_read, buff, sizeof(buff)) < 0) {
            //     perror("Error when reading ack");
            // }
            // std::cout << "AFTER READING from FIFO PIPE: " << buff[0] << std::endl;
            // close(fifo_read);
        }
        box_vector.clear();
        coords.clear();
        coordsGeo.clear();
        boxCoords.clear();

        if (n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;


        frameAmount += n_batch;
	iters++;
    }

    std::cout<<"detection end\n";
    /*if (use_socket) {
        // sending flag to python workflow to mark the end of the video processing
        char flag = 0;
        app_socket->send(&flag, 0);
        app_socket->recv(&unimportant_message);
        // if (write(fifo_write, &flag, 0) < 0) {
        //     perror("Error when writing");
        // }
    }*/

    double mean = 0;
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for (int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;

    if (use_socket) {
        if (app_socket->connected()) {
            app_socket->close();
        }
        delete app_socket;
    }
    return 0;
}
