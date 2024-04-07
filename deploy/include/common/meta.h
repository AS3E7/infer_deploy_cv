#ifndef META_H
#define META_H

namespace gddeploy{
/*
存储结果

层次关系：
BatchMeta->frameMeta->objectMeta->classifierMeta/detectMeta/labelmeta
*/


/** Defines the number of additional fields available to the user
 in the metadata structure. */
#define MAX_USER_FIELDS 4

#define MAX_RESERVED_FIELDS 4
/** Defines the maximum size of an array for storing a text result. */
#define MAX_LABEL_SIZE 128
/** Defines the maximum number of elements that a given display meta
 can hold. */
#define MAX_ELEMENTS_IN_DISPLAY_META 16
/**
 * Holds information about a formed batch containing frames from different
 * sources.
 * NOTE: Both Video and Audio metadata uses the same NvDsBatchMeta type.
 * NOTE: Audio batch metadata is formed within nvinferaudio plugin
 * and will not be corresponding to any one buffer output from nvinferaudio.
 * The NvDsBatchMeta for audio is attached to the last input buffer
 * when the audio batch buffering reach configurable threshold
 * (audio frame length) and this is when inference output is available.
 */
typedef struct _NvDsBatchMeta {
  NvDsBaseMeta base_meta;
  /** Holds the maximum number of frames in the batch. */
  uint32_t max_frames_in_batch;
  /** Holds the number of frames now in the batch. */
  uint32_t num_frames_in_batch;
  /** Holds a pointer to a pool of pointers of type @ref NvDsFrameMeta,
   representing a pool of frame metas. */
  NvDsMetaPool *frame_meta_pool;
  /** Holds a pointer to a pool of pointers of type NvDsObjMeta,
   representing a pool of object metas. */
  NvDsMetaPool *obj_meta_pool;
  /** Holds a pointer to a pool of pointers of type @ref NvDsClassifierMeta,
   representing a pool of classifier metas. */
  NvDsMetaPool *classifier_meta_pool;
  /** Holds a pointer to a pool of pointers of type @ref NvDsDisplayMeta,
   representing a pool of display metas. */
  NvDsMetaPool *display_meta_pool;
  /** Holds a pointer to a pool of pointers of type @ref NvDsUserMeta,
   representing a pool of user metas. */
  NvDsMetaPool *user_meta_pool;
  /** Holds a pointer to a pool of pointers of type @ref NvDsLabelInfo,
   representing a pool of label metas. */
  NvDsMetaPool *label_info_meta_pool;
  /** Holds a pointer to a list of pointers of type NvDsFrameMeta
   or NvDsAudioFrameMeta (when the batch represent audio batch),
   representing frame metas used in the current batch.
   */
  NvDsFrameMetaList *frame_meta_list;
  /** Holds a pointer to a list of pointers of type NvDsUserMeta,
   representing user metas in the current batch. */
  NvDsUserMetaList *batch_user_meta_list;
  /** Holds a lock to be set before accessing metadata to avoid
   simultaneous update by multiple components. */
  GRecMutex meta_mutex;
  /** Holds an array of user-specific batch information. */
  uint64_t misc_batch_info[MAX_USER_FIELDS];
  /** For internal use. */
  uint64_t reserved[MAX_RESERVED_FIELDS];
} NvDsBatchMeta;

/**
 * Holds metadata for a frame in a batch.
 */
typedef struct _NvDsFrameMeta {
  /** Holds the base metadata for the frame. */
  NvDsBaseMeta base_meta;
  /** Holds the pad or port index of the Gst-streammux plugin for the frame
   in the batch. */
  uint32_t pad_index;
  /** Holds the location of the frame in the batch. The frame's
   @ref NvBufSurfaceParams are at index @a batch_id in the @a surfaceList
   array of @ref NvBufSurface. */
  uint32_t batch_id;
  /** Holds the current frame number of the source. */
  int32_t frame_num;
  /** Holds the presentation timestamp (PTS) of the frame. */
  guint64 buf_pts;
  /** Holds the ntp timestamp. */
  guint64 ntp_timestamp;
  /** Holds the source IDof the frame in the batch, e.g. the camera ID.
   It need not be in sequential order. */
  uint32_t source_id;
  /** Holds the number of surfaces in the frame, required in case of
   multiple surfaces in the frame. */
  int32_t num_surfaces_per_frame;
  /* Holds the width of the frame at input to Gst-streammux. */
  uint32_t source_frame_width;
  /* Holds the height of the frame at input to Gst-streammux. */
  uint32_t source_frame_height;
  /* Holds the surface type of the subframe, required in case of
   multiple surfaces in the frame. */
  uint32_t surface_type;
  /* Holds the surface index of tje subframe, required in case of
   multiple surfaces in the frame. */
  uint32_t surface_index;
  /** Holds the number of object meta elements attached to current frame. */
  uint32_t num_obj_meta;
  /** Holds a Boolean indicating whether inference is performed on the frame. */
  gboolean bInferDone;
  /** Holds a pointer to a list of pointers of type @ref NvDsObjectMeta
   in use for the frame. */
  NvDsObjectMetaList *obj_meta_list;
  /** Holds a pointer to a list of pointers of type @ref NvDsDisplayMeta
   in use for the frame. */
  NvDisplayMetaList *display_meta_list;
  /** Holds a pointer to a list of pointers of type @ref NvDsUserMeta
   in use for the frame. */
  NvDsUserMetaList *frame_user_meta_list;
  /** Holds additional user-defined frame information. */
  uint64_t misc_frame_info[MAX_USER_FIELDS];
  /** For internal use. */
  uint64_t reserved[MAX_RESERVED_FIELDS];
} NvDsFrameMeta;

/**
 * Holds metadata for an object in the frame.
 */
typedef struct _NvDsObjectMeta {
  NvDsBaseMeta base_meta;
  /** Holds a pointer to the parent @ref NvDsObjectMeta. Set to NULL if
   no parent exists. */
  struct _NvDsObjectMeta *parent;
  /** Holds a unique component ID that identifies the metadata
   in this structure. */
  int32_t unique_component_id;
  /** Holds the index of the object class inferred by the primary
   detector/classifier. */
  int32_t class_id;
  /** Holds a unique ID for tracking the object. @ref UNTRACKED_OBJECT_ID
   indicates that the object has not been tracked. */
  guint64 object_id;
  /** Holds a structure containing bounding box parameters of the object when
    detected by detector. */
  NvDsComp_BboxInfo detector_bbox_info;
  /** Holds a structure containing bounding box coordinates of the object when
   * processed by tracker. */
  NvDsComp_BboxInfo tracker_bbox_info;
  /** Holds a confidence value for the object, set by the inference
   component. confidence will be set to -0.1, if "Group Rectangles" mode of
   clustering is chosen since the algorithm does not preserve confidence
   values. Also, for objects found by tracker and not inference component,
   confidence will be set to -0.1 */
  float confidence;
  /** Holds a confidence value for the object set by nvdcf_tracker.
   * tracker_confidence will be set to -0.1 for KLT and IOU tracker */
  float tracker_confidence;
  /** Holds a structure containing positional parameters of the object
   * processed by the last component that updates it in the pipeline.
   * e.g. If the tracker component is after the detector component in the
   * pipeline then positinal parameters are from tracker component.
   * Positional parameters are clipped so that they do not fall outside frame
   * boundary. Can also be used to overlay borders or semi-transparent boxes on
   * objects. @see NvOSD_RectParams. */
  NvOSD_RectParams rect_params;
  /** Holds mask parameters for the object. This mask is overlayed on object
   * @see NvOSD_MaskParams. */
  NvOSD_MaskParams mask_params;
  /** Holds text describing the object. This text can be overlayed on the
   standard text that identifies the object. @see NvOSD_TextParams. */
  NvOSD_TextParams text_params;
  /** Holds a string describing the class of the detected object. */
  int8_t obj_label[MAX_LABEL_SIZE];
  /** Holds a pointer to a list of pointers of type @ref NvDsClassifierMeta. */
  NvDsClassifierMetaList *classifier_meta_list;
  /** Holds a pointer to a list of pointers of type @ref NvDsUserMeta. */
  NvDsUserMetaList *obj_user_meta_list;
  /** Holds additional user-defined object information. */
  uint64_t misc_obj_info[MAX_USER_FIELDS];
  /** For internal use. */
  uint64_t reserved[MAX_RESERVED_FIELDS];
}NvDsObjectMeta;

/**
 * Holds classifier metadata for an object.
 */
typedef struct _NvDsClassifierMeta {
  NvDsBaseMeta base_meta;
  /** Holds the number of outputs/labels produced by the classifier. */
  uint32_t num_labels;
  /** Holds a unique component ID for the classifier metadata. */
  int32_t unique_component_id;
  /** Holds a pointer to a list of pointers of type @ref NvDsLabelInfo. */
  NvDsLabelInfoList *label_info_list;
} NvDsClassifierMeta;

/**
 * Holds label metadata for the classifier.
 */
typedef struct _NvDsLabelInfo {
  NvDsBaseMeta base_meta;
  /** Holds the number of classes of the given label. */
  uint32_t num_classes;
  /** Holds an string describing the label of the classified object. */
  int8_t result_label[MAX_LABEL_SIZE];
  /** Holds a pointer to the result label if its length exceeds MAX_LABEL_SIZE bytes. */
  int8_t *pResult_label;
  /** Holds the class UD of the best result. */
  uint32_t result_class_id;
  /** Holds the label ID in case there are multiple label classifiers. */
  uint32_t label_id;
  /** Holds the probability of best result. */
  float result_prob;
} NvDsLabelInfo;

}

#endif