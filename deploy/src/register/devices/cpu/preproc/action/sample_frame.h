#pragma once

#include <vector>

class UniformSampleFrames
{
public:
    UniformSampleFrames(){}
    UniformSampleFrames(int clip_len, int num_clips, bool test_mode, int seed);
    ~UniformSampleFrames();

    std::vector<int> sample_frames(int num_frames);
private:
    int clip_len;
    int num_clips;
    int seed;
    bool test_mode;
};