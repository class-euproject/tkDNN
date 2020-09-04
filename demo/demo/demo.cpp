#include <iostream>
#include <csignal>
#include <stdlib.h>     /* srand, rand */
#include <mutex>
#include <data.h>
#include <string>
#include <camera_elaboration.h>
#include <configuration.h>
#include "zmq.hpp"

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

bool gRun;
bool SAVE_RESULT = false;

zmq::socket_t *app_socket;

void sig_handler(int signo) {
    std::cout<<"request gateway stop\n";
    gRun = false;
    app_socket->close();
}

edge::camera prepareCamera(int camera_id, std::string &net, char &type, int &n_classes) {
    YAML::Node config = YAML::LoadFile("../../data/all_cameras_en.yaml");
    YAML::Node cameras_yaml = config["cameras"];

    net = config["net"].as<std::string>();
    type = config["type"].as<char>();
    n_classes = config["classes"].as<int>();
    std::string tif_map_path = config["tif"].as<std::string>();

    edge::camera_params camera_par;
    for (auto && cam_yaml : cameras_yaml) {
        int ref_cam_id = cam_yaml["id"].as<int>();
        if (ref_cam_id != camera_id) continue;
        if (cam_yaml["encrypted"].as<int>()) {
            //camera_par.input = decryptString(cam_yaml["input"].as<std::string>(),);
            throw;
        } else {
            camera_par.input = cam_yaml["input"].as<std::string>();
        }
        camera_par.pmatrixPath        = cam_yaml["pmatrix"].as<std::string>();
        camera_par.maskfilePath       = cam_yaml["maskfile"].as<std::string>();
        camera_par.cameraCalibPath    = cam_yaml["cameraCalib"].as<std::string>();
        camera_par.maskFileOrientPath = cam_yaml["maskFileOrient"].as<std::string>();
        break;
    }

    std::cout << "Camera parameters read!" << std::endl << camera_par << std::endl;

    edge::Dataset_t dataset;
    switch (n_classes) {
        case 10: dataset = edge::Dataset_t::BDD; break;
        case 80: dataset = edge::Dataset_t::COCO; break;
        default: FatalError("Dataset type not supported yet, check number of classes in parameter file.");
    }

    edge::camera camera;
    std::cout << "Reading calibration matrix in " << camera_par.cameraCalibPath << std::endl;
    readCalibrationMatrix(camera_par.cameraCalibPath, camera.calibMat, camera.distCoeff, camera.calibWidth, camera.calibHeight);
    std::cout << "Calibration matrix read!" << std::endl << camera.calibMat << std::endl;
    std::cout << "Reading projection matrix in " << camera_par.pmatrixPath << std::endl;
    readProjectionMatrix(camera_par.pmatrixPath, camera.prjMat);
    std::cout << "Projection matrix read!" << std::endl << camera.calibMat << std::endl;
    camera.id = camera_par.id;
    camera.input = camera_par.input;
    camera.streamWidth = config["width"].as<int>();
    camera.streamHeight = config["height"].as<int>();
    camera.show = true; // THIS SHOULD BE INPUT
    camera.invPrjMat = camera.prjMat.inv();
    camera.dataset = dataset;

    camera.adfGeoTransform = (double *) malloc(6 * sizeof(double));
    readTiff(tif_map_path, camera.adfGeoTransform);
    camera.geoConv.initialiseReference(44.655540, 10.934315, 0); // THIS?
    return camera;
}

char* prepareMessage(std::vector<tk::dnn::box> &box_vector, std::vector<std::tuple<float, float>> &coords, unsigned int frameAmount, unsigned int *size) {
    *size = box_vector.size() * (sizeof(double) * 2 + sizeof(int) + 1 + sizeof(float) * 4) + 1;
    char *data = (char *) malloc(*size);
    char *data_origin = data;
    char flag = ~0;
    memcpy(data++, &flag, 1);
    /*
    char double_size = (char) sizeof(double);
    memcpy(data++, &double_size, 1);
    char int_size = (char) sizeof(int);
    memcpy(data++, &int_size, 1);
    char float_size = (char) sizeof(float);
    memcpy(data++, &float_size, 1);
     */
    for (int i = 0; i < box_vector.size(); i++) {
        tk::dnn::box box = box_vector[i];
        std::tuple<float, float> coord = coords[i];
        double coord_north = std::get<0>(coord);
        double coord_east = std::get<1>(coord);
        // double double uint char float float float float
        // printf("%f %f %u %i %f %f %f %f\n", coord_north, coord_east, frameAmount, box.cl, box.x, box.y, box.w, box.h);
        memcpy(data, &coord_north, sizeof(double));
        data += sizeof(double);
        memcpy(data, &coord_east, sizeof(double));
        data += sizeof(double);
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
    }
    // printf("\n");
    return data_origin;
}

int main(int argc, char *argv[]) {

    std::cout<<"detection!\n";
    signal(SIGINT, sig_handler);

    int camera_id = 20939;
    if (argc > 1) {
        camera_id = atoi(argv[1]);
    }
    std::string socketPort = "5559";
    if(argc > 2)
        socketPort = argv[1];
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
    if(argc > 3)
        show = atoi(argv[3]);
    int n_batch = 1;
    if(argc > 4)
        n_batch = atoi(argv[4]);

    std::string net;
    char ntype;
    int n_classes;

    edge::camera camera = prepareCamera(camera_id, net, ntype, n_classes);

    if(n_batch < 1 || n_batch > 64)
        FatalError("Batch dim not supported");

    if(!show)
        SAVE_RESULT = true;

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
    if(show)
        cv::namedWindow("detection", cv::WINDOW_NORMAL);

    std::vector<cv::Mat> batch_frame;
    std::vector<cv::Mat> batch_dnn_input;

    zmq::context_t context(1);
    zmq::message_t unimportant_message;
    app_socket = new zmq::socket_t(context, ZMQ_REQ);
    app_socket->bind("tcp://0.0.0.0:" + socketPort);

    unsigned int frameAmount = 0;
    std::vector<tk::dnn::box> box_vector;
    std::vector<std::tuple<float, float>> coords;
    while(gRun) {
        batch_dnn_input.clear();
        batch_frame.clear();
        
        for(int bi=0; bi< n_batch; ++bi){
            cap >> frame; 
            if(!frame.data)
                break;
            
            batch_frame.push_back(frame);

            // this will be resized to the net format
            batch_dnn_input.push_back(frame.clone());
        }
        if(!frame.data) 
            break;
    
        //inference
        detNN->update(batch_dnn_input, n_batch);
        for (auto &box_batch : detNN->batchDetected) {
            for (auto &box : box_batch) {
                convertCameraPixelsToMapMeters(box.x, box.y, box.cl, camera, north, east);
                // pixel2GPS(box.x, box.y, lat, lon, camera.adfGeoTransform);
                box_vector.push_back(box);
                coords.push_back(std::make_tuple(north, east));
                // printf("\t(%f,%f) converted to (%f,%f)\n", box.x, box.y, north, east);
            }
        }
        detNN->draw(batch_frame);

        // send thru socket
        unsigned int size;
        char *data = prepareMessage(box_vector, coords, frameAmount, &size);
        zmq::message_t message(size);
        memcpy(message.data(), data, size);
        std::cout << "[" << frameAmount << "] Waiting for message..." << std::endl;
        app_socket->send(message, 0);
        free(data);
        app_socket->recv(&unimportant_message);
        box_vector.clear();
        coords.clear();

        if (show) {
            for (int bi=0; bi < n_batch; ++bi) {
                cv::imshow("detection", batch_frame[bi]);
                cv::waitKey(1);
            }
        }
        /*
        if (n_batch == 1 && SAVE_RESULT)
            resultVideo << frame;*/

        frameAmount += n_batch;
    }

    std::cout<<"detection end\n";
    char flag = 0;
    app_socket->send(&flag, 0);
    app_socket->recv(&unimportant_message);

    double mean = 0;
    std::cout<<COL_GREENB<<"\n\nTime stats:\n";
    std::cout<<"Min: "<<*std::min_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";
    std::cout<<"Max: "<<*std::max_element(detNN->stats.begin(), detNN->stats.end())/n_batch<<" ms\n";    
    for (int i=0; i<detNN->stats.size(); i++) mean += detNN->stats[i]; mean /= detNN->stats.size();
    std::cout<<"Avg: "<<mean/n_batch<<" ms\t"<<1000/(mean/n_batch)<<" FPS\n"<<COL_END;

    if (app_socket->connected()) {
        app_socket->close();
    }
    delete app_socket;

    return 0;
}