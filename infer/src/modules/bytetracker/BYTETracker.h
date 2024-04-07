#pragma once

#include"Track.h"


struct TargetObject
{
	int box_id;
	int label;
	float prob;
	cv::Rect_<float> rect;
};


class BYTETracker
{
public:
/************************************************************************************
	参数:
        det_thresh (float): 检测分数大于det_thresh才会建立跟踪框。
		unconfirmed_thresh (float): iou Distance大于该阈值会进行重新匹配。
		conf_thresh (float)：检测分数大于该阈值的框进入高分队列。
		low_conf_thresh (float):检测分数大于low_conf_thres并小于conf_thres的框进入低分队列，默认值0.1。
		match_thresh (float):iou Distance大于该阈值会直接匹配上。
		low_match_thresh (float): 使用低分框匹配的iou Distance阈值。
		max_frame_lost (int): 最大丢失帧数。
		n_classes(int):分类数目，1分类填1。
	说明:
		1、det_thresh > conf_thresh > low_conf_thresh。
		2、match_thresh > unconfirmed_thresh > low_match_thresh。
****************************************************************************************/

	BYTETracker(
		const float det_thresh,
		const float unconfirmed_thresh,
		const float conf_thresh,
		const float low_conf_thresh,
		const float match_thresh,
		const float low_match_thresh,
		const int max_frame_lost,
		const int n_classes);

	~BYTETracker();

	vector<Track> update(const vector<TargetObject>& objects);
	unordered_map<int, vector<Track>> updateMCMOT(const vector<TargetObject>& objects);

	static Scalar getColor(int idx);
	// tracking object class number
	int m_N_CLASSES;
private:
	vector<Track*> joinTracks(vector<Track*>& tlista,
		vector<Track>& tlistb);

	vector<Track> joinTracks(vector<Track>& tlista,
		vector<Track>& tlistb);

	vector<Track> subTracks(vector<Track>& tlista,
		vector<Track>& tlistb);

	void removeDuplicateTracks(vector<Track>& resa,
		vector<Track>& resb,
		vector<Track>& tracks_a,
		vector<Track>& tracks_b);

	void linearAssignment(vector<vector<float>>& cost_matrix, 
		int cost_matrix_size, 
		int cost_matrix_size_size,
		float thresh,
		vector<vector<int>>& matches,
		vector<int>& unmatched_a,
		vector<int>& unmatched_b);

	vector<vector<float>> iouDistance(vector<Track*>& atracks,
		vector<Track>& btracks, 
		int& dist_size,
		int& dist_size_size);

	vector<vector<float>> iouDistance(vector<Track>& atracks, 
		vector<Track>& btracks);

	vector<vector<float>> ious(vector<vector<float>>& atlbrs,
		vector<vector<float>>& btlbrs);

	double lapjv(const vector<vector<float>>& cost,
		vector<int>& rowsol, 
		vector<int>& colsol,
		bool extend_cost = false, 
		float cost_limit = LONG_MAX, 
		bool return_cost = true);

private:

	float m_high_det_thresh;
	float m_low_det_thresh;
	float m_new_track_thresh;
	float m_high_match_thresh;
	float m_low_match_thresh;
	float m_unconfirmed_match_thresh;
	int m_frame_id;
	int m_max_time_lost;

	// 3 containers of the tracker
	vector<Track> m_tracked_tracks;
	vector<Track> m_lost_tracks;
	vector<Track> m_removed_tracks;

	unordered_map<int, vector<Track>> m_tracked_tracks_dict;
	unordered_map<int, vector<Track>> m_lost_tracks_dict;
	unordered_map<int, vector<Track>> m_removed_tracks_dict;

	byte_kalman::KalmanFilter m_kalman_filter;
};