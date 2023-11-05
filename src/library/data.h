#ifndef DATA_H
#define DATA_H
typedef struct {
    int width;
    int height;
    int channels;
    float *data;
} ImgData;

ImgData* load_img_data(char *path);
void free_img_data(ImgData *ImgData);
#endif