#include "target_tracker.h"
#include "BYTETracker.h"
#include <memory>

namespace gddi {

class TargetTrackerPrivate {
public:
    TargetTrackerPrivate(const float track_thres, const float high_thresh, const float match_thresh,
                         const int max_frame_lost) {
        tracker_ =
            std::make_unique<BYTETracker>(track_thres, 0.7, high_thresh, 0.1, match_thresh, 0.4, max_frame_lost, 10);
    }

    std::map<int, TrackObject> update_object(const std::vector<TrackObject> &objects) {

        auto vec_objects = std::vector<TargetObject>();
        for (const auto &item : objects) {
            vec_objects.push_back(TargetObject{.box_id = item.target_id,
                                               .label = item.class_id,
                                               .prob = item.prob,
                                               .rect = {item.rect.x, item.rect.y, item.rect.width, item.rect.height}});
        }

        std::map<int, TrackObject> result;
        auto tracked_target = tracker_->updateMCMOT(vec_objects);
        for (auto &target : tracked_target) {
            for (auto &item : target.second) {
                result[item.track_id] =
                    TrackObject{.target_id = item.box_id,
                                .class_id = item.class_id,
                                .prob = item.score,
                                .rect = Rect2f{item.tlwh[0], item.tlwh[1], item.tlwh[2], item.tlwh[3]}};
            }
        }

        return result;
    }

    ~TargetTrackerPrivate() {}

private:
    std::unique_ptr<BYTETracker> tracker_;
};

TargetTracker::TargetTracker(const TrackOption &option) {
    uq_impl_ = std::make_unique<TargetTrackerPrivate>(option.track_thresh, option.high_thresh, option.match_thresh,
                                                      option.max_frame_lost);
}

TargetTracker::~TargetTracker() {}

std::map<int, TrackObject> TargetTracker::update_objects(const std::vector<TrackObject> &objects) {
    return uq_impl_->update_object(objects);
}

}// namespace gddi