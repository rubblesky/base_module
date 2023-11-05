#include <stdio.h>
#include <stdlib.h>
#include "data.h"
ImgData* load_img_data(char *path) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        printf("cannot open the file\n");
        exit(-1);
    }
    // 读取宽度、高度和通道数
    ImgData * img_data = malloc(sizeof(ImgData));
    size_t size = 0;
    size += fread(&img_data->width, sizeof(int), 1, file);
    size +=fread(&img_data->height, sizeof(int), 1, file);
    size +=fread(&img_data->channels, sizeof(int), 1, file);
    if(size != 3){
        printf("error\n");
        exit(-1);
    }
    float * image = (float *)malloc(img_data->width * img_data->height * img_data->channels * sizeof(float));
    img_data->data = image;
    size = fread(image, sizeof(int), img_data->width * img_data->height * img_data->channels, file);
    printf("size = %d width * height * channels = %d\n", size,img_data->width * img_data->height * img_data->channels);
    fclose(file);
    return img_data;
}
void free_img_data(ImgData *image) {  
    free(image->data);
    free(image);
}
