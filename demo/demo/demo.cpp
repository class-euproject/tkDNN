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
#include <netinet/in.h>

#include <sys/types.h>
#include <sys/socket.h>

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;

edge::camera camera;
bool limitedIters;
int num_iters;
int iters;
cv::VideoCapture cap;
struct video_cap_data
{
    char* input         = nullptr;
    cv::Mat frame;
    uint64_t tStampMs;
    std::mutex mtxF;
    int width           = 0;
    int height          = 0;
    int camId           = 0;
    bool frameConsumed  = false;
};

#define PORT 5559
int sockfd = -1;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
    if (sockfd != -1) {
        close(sockfd);
    }
    FatalError("Closing application");
}

void *readVideoCapture( void *ptr )
{
    video_cap_data* data = (video_cap_data*) ptr;
    const int new_width     = data->width;
    const int new_height    = data->height;
    cv::Mat frame, resized_frame;
    cv::VideoCapture cap(camera.input, cv::CAP_FFMPEG);
    if(!cap.isOpened()) {
        std::cout << "Camera could not be started." << std::endl;
        exit(1);
    } else {
        std::cout << "Camera started" << std::endl;
    }
    cv::VideoWriter result_video;
    bool record = true;
    if (record){
        int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        result_video.open("video_cam_"+std::to_string(data->camId)+".mp4", cv::VideoWriter::fourcc('M','P','4','V'), 30, cv::Size(w, h));
    }
    bool retval;
    while(not limitedIters or iters < num_iters) {
        if(!data->frameConsumed) {
            usleep(1000);
            continue;
        }
        retval = cap.read(frame);
        if (!retval) {
            std::cout << "Error when reading frame from stream. Retrying." << std::endl;
            cap.open(camera.input);
            continue;
        }
        // resizing the image to 960x540 (the DNN takes in input 544x320)
        cv::resize (frame, resized_frame, cv::Size(new_width, new_height));
        data->mtxF.lock();
        data->frame         = resized_frame.clone();
        data->frameConsumed = false;
        data->mtxF.unlock();
        result_video << frame;
    }
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
            /*std::cout << "The input file is encrypted. Throwing exception" << std::endl;
            throw;*/
        } else {
            camera_par.input = cam_yaml["input"].as<std::string>();
        }
        if(cam_yaml["resolution"]) {
            camera_par.resolution  = cam_yaml["resolution"].as<std::string>();
            // here we append some parameters to the rtsp string
            if(!camera_par.resolution.empty())
                camera_par.input   = camera_par.input+"?resolution="+camera_par.resolution;
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
    camera.show = true;
    camera.hasCalib = true;
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
                     unsigned int frameAmount, int cam_id, double lat_init, double lon_init, unsigned int *size,
                     float scale_x, float scale_y) {
    /*box_vector.erase(std::remove_if(box_vector.begin(), box_vector.end(), [](tk::dnn::box &box) {
        return box.cl == 7 || box.cl == 8;
    }), box_vector.end()); // if traffic signs or traffic lights*/
    for (int i = box_vector.size() - 1; i >= 0; i--) {
        // if traffic signs or traffic lights
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
    float box_x, box_y, box_w, box_h;
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
        box_x = box.x * scale_x;
        memcpy(data, &box_x, sizeof(float));
        data += sizeof(float);
        box_y = box.y * scale_y;
        memcpy(data, &box_y, sizeof(float));
        data += sizeof(float);
        box_w = box.w * scale_x;
        memcpy(data, &box_w, sizeof(float));
        data += sizeof(float);
        box_h = box.h * scale_y;
        memcpy(data, &box_h, sizeof(float));
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
    return data_origin;
}

int main(int argc, char *argv[]) {
    std::cout<<"detection!\n";
    signal(SIGINT, sig_handler);

    bool use_socket = true;
    if (argc > 1) {
        use_socket = atoi(argv[1]);
    }
    int camera_id = 20939;
    if (argc > 2) {
        camera_id = atoi(argv[2]);
    }
    std::string socketPort = "5559";
	int port = 5559;

    int argv_ref = 2;
    if (use_socket) {
        if (argc > ++argv_ref) {
            socketPort = argv[argv_ref];
            port = atoi(argv[argv_ref]);
        }
    
    }

    num_iters = -1;
    limitedIters = false;
    if (argc > ++argv_ref) {
        num_iters = atoi(argv[argv_ref]);
        limitedIters = (num_iters != -1);
    }

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

    camera = prepareCamera(camera_id, net, ntype, n_classes, show);

    if (n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

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

    struct sockaddr_in servaddr;

    if (use_socket) {
        std::cout << "Listening to tcp://0.0.0.0:" << socketPort << " waiting for COMPSs ack to start" << std::endl;
        zmq::socket_t *app_socket = new zmq::socket_t(context, ZMQ_REP);
        app_socket->bind("tcp://0.0.0.0:" + socketPort);
        app_socket->recv(&unimportant_message); // wait for python workflow to ack to start processing frames
        app_socket->close();
        delete app_socket;

        std::cout << "Listening to udp in://0.0.0.0:" << socketPort << std::endl;

	    // Creating socket file descriptor
	    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
		    perror("socket creation failed");
		    exit(EXIT_FAILURE);
	    }

	    servaddr.sin_family = AF_INET;
	    servaddr.sin_port = htons(port);
	    servaddr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
        {
            std::cout << "Failed to bind socket! " << strerror(errno) << "\n";
            return 1;
        }
    } else {
        std::cout << "Not opening socket" << std::endl;
    }

    pthread_t video_cap;
    video_cap_data data;
    data.input          = (char*)camera.input.c_str();
    data.width          = camera.streamWidth;
    data.height         = camera.streamHeight;
    data.camId          = camera.id;
    data.frameConsumed  = true;
    if (pthread_create(&video_cap, NULL, readVideoCapture, (void *)&data)){
        fprintf(stderr, "Error creating thread\n");
    }

    unsigned int frameAmount = 0;
    float scale_x   = camera.hasCalib ? (float) camera.calibWidth  / (float) camera.streamWidth : 1;
    float scale_y   = camera.hasCalib ? (float) camera.calibHeight / (float) camera.streamHeight: 1;
    std::vector<tk::dnn::box> box_vector;
    std::vector<std::tuple<double, double>> coords;
    std::vector<std::tuple<double, double>> coordsGeo;
    std::vector<std::tuple<double, double, double, double, double, double, double, double>> boxCoords;
    double lat_ur, lat_lr, lat_ll, lat_ul, lon_ur, lon_lr, lon_ll, lon_ul;
    int iters = 0;
    bool new_frame = false;
    while(not limitedIters or iters < num_iters) {
        batch_dnn_input.clear();
        batch_frame.clear();
        data.mtxF.lock();
        new_frame = data.frameConsumed;
        frame = data.frame;
        data.frameConsumed = true;
        data.mtxF.unlock();
        if (new_frame) {
            usleep(1000);
            continue;
        }
        batch_frame.push_back(frame);
        // this will be resized to the net format
        batch_dnn_input.push_back(frame.clone());

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
                boxCoords.push_back(std::make_tuple(lat_ur, lon_ur, lat_lr, lon_lr, lat_ll, lon_ll, lat_ul, lon_ul));
            }
        }


        // send thru socket or pipe
        if (use_socket) {
            unsigned int size;
	        char *data = prepareMessage(box_vector, coords, boxCoords, frameAmount, camera_id,
	                camera.adfGeoTransform[3], camera.adfGeoTransform[0], &size, scale_x, scale_y);


            std::cout << "[" << frameAmount << "] Processing frame..." << std::endl;

	        char clientname[1024] = "";
	        struct sockaddr_in clientaddr = sockaddr_in();
	        socklen_t len = sizeof(clientaddr);

            inet_ntop(AF_INET,&clientaddr.sin_addr,clientname,sizeof(clientname));
	        std::string client_ip_str(clientname);

	        std::cout << "Client ip address: " << client_ip_str << std::endl;

	        char buffer_recv[1024];
            int recv_error = recvfrom(sockfd, buffer_recv, 1024,
                MSG_DONTWAIT, ( struct sockaddr *) &clientaddr,
                &len);

            inet_ntop(AF_INET,&clientaddr.sin_addr,clientname,sizeof(clientname));
            client_ip_str = std::string(clientname);

	        std::cout << "Client ip address after readfrom: " << client_ip_str << std::endl;

	        if (client_ip_str != "0.0.0.0"){
		        std::cout << "Sending data to " << client_ip_str << std::endl;
            	sendto(sockfd, (const char *)data, size,MSG_DONTWAIT, (const struct sockaddr *) &clientaddr,sizeof(clientaddr));
            }

            free(data); 
     
            //n = recvfrom(sockfd, (char *)recvBuffer, 1024, MSG_WAITALL,
            //     (struct sockaddr *) &servaddr,
            //       &len);
            //ecvBuffer[n] = '\0';

        }
        box_vector.clear();
        coords.clear();
        coordsGeo.clear();
        boxCoords.clear();

        detNN->draw(batch_frame);

        if (show) {
            for (int bi=0; bi < n_batch; ++bi) {
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }

        frameAmount += n_batch;
        iters++;
    }

    pthread_join( video_cap, NULL);

    std::cout<<"detection end\n";

    double mean = 0;
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for (int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;

    if (use_socket) {
		close(sockfd);
    }
    return 0;
}

