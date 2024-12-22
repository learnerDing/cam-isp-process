#ifndef RAW10TORAW8_H
#define RAW10TORAW8_H

#include "Isppipeline.h"

class Raw10ToRaw8 : public ISPPipeline {
public:
    void process(const Image& input, Image& output) override;
};

#endif // RAW10TORAW8_H