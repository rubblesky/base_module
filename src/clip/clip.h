#ifndef CLIP_H
#define CLIP_H
#include "vision.h"

typedef struct LanguageEncoder LanguageEncoder;
typedef struct CLIP{
    VisionEncoder * ve;
    LanguageEncoder * le;
}CLIP;

#endif
