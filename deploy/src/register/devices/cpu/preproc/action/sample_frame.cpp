#include "sample_frame.h"
#include <iostream>
#include <random>

// template for random choice
template <typename T>
T random_choice(std::vector<T> &vec)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, vec.size() - 1);
    return vec[dis(gen)];
}

// template for arange function
template <typename T>
std::vector<T> arange(T start, T end, T step = 1)
{
    std::vector<T> v;
    for (T i = start; i < end; i += step)
        v.push_back(i);
    return v;
}

// template for zero function
template <typename T>
std::vector<T> zeros(T size)
{
    std::vector<T> v;
    for (int i = 0; i < size; i++)
        v.push_back(0);
    return v;
}

// template for cumsum function
template <typename T>
std::vector<T> cumsum(std::vector<T> &v)
{
    std::vector<T> res;
    T sum = 0;
    for (auto i : v)
    {
        sum += i;
        res.push_back(sum);
    }
    return res;
}

// template for concatenate function
template <typename T>
std::vector<T> concatenate(std::vector<T> &v1, std::vector<T> &v2)
{
    std::vector<T> res;
    res.insert(res.end(), v1.begin(), v1.end());
    res.insert(res.end(), v2.begin(), v2.end());
    return res;
}

// template for diff function
template <typename T>
std::vector<T> diff(std::vector<T> &v)
{
    std::vector<T> res;
    for (unsigned int i = 1; i < v.size(); i++)
        res.push_back(v[i] - v[i - 1]);
    return res;
}

// template for randint function
template <typename T>
std::vector<T> randint(T start, T end, T size)
{
    std::vector<T> v;
    for (int i = 0; i < size; i++)
        v.push_back(rand() % (end - start) + start);
    return v;
}

UniformSampleFrames::UniformSampleFrames(int clip_len, int num_clips, bool test_mode, int seed)
{
    this->clip_len = clip_len;
    this->num_clips = num_clips;
    this->test_mode = test_mode;
    this->seed = seed;
}

UniformSampleFrames::~UniformSampleFrames()
{
}

/**
 * @brief sample an n-frame clip from the video. UniformSampleFrames basically
 * divide the video into n segments of equal length and randomly sample one
 * frame from each segment. To make the testing results reproducible, a
 * random seed is set during testing, to make the sampling results
 * deterministic.
 *
 * @param clip_len (int): Frames of each sampled output clip.
 * @param num_clips (int): Number of clips to be sampled. Default: 1.
 * @param description (bool): Store True when building test or validation dataset. Default: False.
 * @param description (int): The random seed used during test time. Default: 255.
 * @return std::vector<int>
 */
std::vector<int> UniformSampleFrames::sample_frames(int num_frames)
{
    std::vector<int> inds;
    std::vector<int> start_inds;

    // std::cout << "UniformSampleFrames::sample_frames: num_frames: " << num_frames << std::endl;
    // std::cout << "clip_len: " << clip_len << std::endl;
    // std::cout << "num_clips: " << num_clips << std::endl;
    // std::cout << "test_mode: " << test_mode << std::endl;
    // std::cout << "seed: " << seed << std::endl;
    // std::cout << "----------------------------------------" << std::endl;

    if (num_frames < this->clip_len)
    {
        if (num_frames < this->num_clips)
        {
            start_inds = arange(0, num_clips);
        }
        else
        {
            for (int i = 0; i < this->num_clips; i++)
            {
                start_inds.push_back(i * num_frames / num_clips);
            }
        }

        for (unsigned int i = 0; i < start_inds.size(); i++)
        {
            auto tmp_v = arange(start_inds[i], start_inds[i] + this->clip_len);
            for (unsigned int j = 0; j < tmp_v.size(); j++)
            {
                inds.push_back(tmp_v[j]);
            }
        }
    }
    else if (clip_len <= num_frames && num_frames < clip_len * 2)
    {
        std::vector<int> all_inds;
        // std::cout << "sample case 2: " << std::endl;
        for (int i = 0; i < num_clips; i++)
        {
            auto basic = arange(0, clip_len);
            auto candidates = arange(0, clip_len + 1);
            for (int j = 0; j < num_frames - clip_len; j++)
            {
                inds.push_back(random_choice(candidates));
            }

            auto offset = zeros(clip_len + 1);
            for (unsigned int i = 0; i < inds.size(); i++)
            {
                offset[inds[i]] = 1;
            }
            auto cumsum_offset = cumsum(offset);

            std::vector<int> tmp_inds;
            for (unsigned int i = 0; i < basic.size(); i++)
            {
                tmp_inds.push_back(basic[i] + cumsum_offset[i]);
            }

            all_inds = concatenate(all_inds, tmp_inds);
        }
        inds = all_inds;
    }
    else
    {
        std::vector<int> bids;
        for (int i = 0; i < clip_len + 1; i++)
        {
            bids.push_back(i * num_frames / clip_len);
            // std::cout << bids[i] << " ";
        }
        auto bsize = diff(bids);
        // std::cout << std::endl;
        // std::cout << "bsize: " << std::endl;
        std::vector<int> bst;
        for (int i = 0; i < clip_len; i++)
        {
            bst.push_back(bids[i]);
        }

        for (int i = 0; i < num_clips; i++)
        {
            std::vector<int> offset;
            for (unsigned int j = 0; j < bsize.size(); j++)
            {
                offset.push_back(randint(0, bsize[j], 1)[0]);
            }
            for (unsigned int j = 0; j < bst.size(); j++)
            {
                inds.push_back(bst[j] + offset[j]);
            }
        }
    }

    // mod of the number of frames
    
    for (unsigned int i = 0; i < inds.size(); i++)
    {
        inds[i] = inds[i] % num_frames;
        // std::cout << inds[i] << " ";
    }
    // std::cout << std::endl;

    // check every indices is valid
    for (unsigned int i = 0; i < inds.size(); i++)
    {
        if (inds[i] < 0 || inds[i] >= num_frames)
        {
            std::cout << "inds[" << i << "]: " << inds[i] << " is invalid" << std::endl;
            exit(1);
        }
    }
    return inds;
}
