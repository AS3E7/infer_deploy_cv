#ifndef _AI_SOLUTION_H_
#define _AI_SOLUTION_H_

#include <stdint.h>

#ifdef __cplusplus__
extern C {
#endif

#define AI_VERSION_MAJOR      (0)
#define AI_VERSION_MINOR      (0)
#define AI_VERSION_MICOR      (0)

/* 结构体大小校验宏 */
#ifndef AI_SIZEOF_CHECK_EQ
#define AI_SIZEOF_CHECK_EQ(dataType,nBytes) \
    typedef char SC_EQ_##dataType[(sizeof(dataType) == (nBytes)) ? 1 : -1];
#endif

/* 指针强制8字节对齐宏 */
#ifndef AI_UNUSED_FORCE_POINTER_ALIGN8
#if defined(_WIN64) || (defined(__WORDSIZE) && __WORDSIZE==64)
#define AI_UNUSED_FORCE_POINTER_ALIGN8(N)
#elif defined(_WIN32) || (defined(__WORDSIZE) && __WORDSIZE==32)
#define AI_UNUSED_FORCE_POINTER_ALIGN8(N) uint32_t unused##N;
#elif defined(__GNUC__)
#define AI_UNUSED_FORCE_POINTER_ALIGN8(N) uint8_t unused##N[8-sizeof(void*)];
#elif defined(CCS)
#define AI_UNUSED_FORCE_POINTER_ALIGN8(N) uint32_t unused##N;
#else
#error "Can't find macro `__WORDSIZE' definition, please specify this macro 32 or 64 base on your platform!"
#endif
#endif

/* 版本号字符串长度 */
#define AI_VERSION_LEN    32

/************************************* 枚举定义 *************************************/

/* 图像颜色空间 */
typedef enum 
{
    AI_CS_YUV420 = 0,                           /* YUV420(I420), 图像U、V分量宽度为Y分量宽度一半，U、V分量高度为Y分量高度一半，
                                                     @n 数据存储格式 YYYY...YYYY, U...U, V...V */
    AI_CS_NV12,                                 /* YUV420SP(NV12)图像U、V分量宽度为Y分量宽度一半，U、V分量高度为Y分量高度一半，
                                                     @n 数据存储格式 YYYY...YYYY, UV...UV */
    AI_CS_NV21,                                 /* YUV420(NV21)_VU 图像U、V分量宽度为Y分量宽度一半，U、V分量高度为Y分量高度一半，
                                                     @n 数据存储格式 YYYY...YYYY, VU...VU */
    AI_CS_Y,                                    /* 灰度图, 图像仅有Y分量数据 */
}ai_colorspace_e;

/* 图像成像类型 */
typedef enum
{
    AI_IMAGELIGHT_UNKNOWN       = 0,
    AI_IMAGELIGHT_VISIBLE       = 1,            /* 可见光图像 */
    AI_IMAGELIGHT_INFRARED      = 2,            /* 红外图像 */
    AI_IMAGELIGHT_THERMOGRAOHY  = 3             /* 热成像图像 */
}ai_image_light_e;


/* 算法支持的工作模式 */
typedef enum
{
    AI_SCHEDULE_SYNC            = (1<<0),       /* 同步模式 */
    AI_SCHEDULE_ASYNC           = (1<<1),       /* 异步模式 */
}ai_schedule_mode_e;

/* 算法输入数据类型 */
typedef enum
{
    AI_INPUT_TYPE_IMAGE,                        /* 图像, 数据结构体类型为 ai_input_image_s */
    AI_INPUT_TYPE_INFO,                         /* 信息数据, 随帧带进来的数据信息，如随帧目标等，数据结构体类型为 ai_input_info_s */
}ai_input_type_e;

/* 目标跟踪状态*/
typedef enum 
{
    AI_OBJ_STATE_INIT           =    0,          /* 无效值   */
    AI_OBJ_STATE_NEW            =    1,          /* 新出现目标 */
    AI_OBJ_STATE_TRACKING       =    2,          /* 目标正常跟踪 */
    AI_OBJ_STATE_HIDDEN         =    3,          /* 目标隐藏，当前未检到该目标，不确定目标一定消失 */
    AI_OBJ_STATE_LOST           =    4           /* 目标消失，目标跟丢后续不会再出现 */
}ai_obj_state_e;

/* 目标运动状态 */
typedef enum 
{
    AI_OBJ_MOTION_STATUS_UNKNOW     = 0,       /* 未知，初始状态*/
    AI_OBJ_MOTION_STATUS_STATIC     = 1,       /* 静止 */
    AI_OBJ_MOTION_STATUS_MOVE       = 2,       /* 运动 */
}ai_obj_motion_status_e;

/* 配置功能枚举 */
typedef enum 
{
    AI_CONFIG_TYPE_UNKNOW           = 0,        /* 无效配置 */
    AI_CONFIG_TYPE_CUSTOM_CONFIG    = 1,        /* 自定义配置，param字段对应json字符串，json格式由协议文档约束 */
    AI_CONFIG_TYPE_SET_RULE         = 2,        /* 下发规则，param字段对应结构体 ai_config_rules_s */
    AI_CONFIG_TYPE_CHANNEL_CONFIG   = 3,        /* 通道配置，param字段对应结构体 ai_config_channel_s */
}ai_config_type_e;

/* 通道配置枚举 */
typedef enum
{
    AI_CHANNEL_CONFIG_TYPE_INVALID  = 0,        /* 无效值 */
    AI_CHANNEL_CONFIG_TYPE_ADD      = 1,        /* 新增通道 */
    AI_CHANNEL_CONFIG_TYPE_DELETE   = 2,        /* 删除通道，释放与该通道相关的所有资源，注意：删除通道前，调用者需确保该通道不再送数据 */
    AI_CHANNEL_CONFIG_TYPE_ENABLE   = 3,        /* 使能通道，该通道可以处理送进来的数据 */
    AI_CHANNEL_CONFIG_TYPE_DISABLE  = 4,        /* 失能通道，该通道不再执行数据处理，但是通道相关的资源不释放 */
}ai_config_chennel_type_e;

/* 注册的回调函数类型 */
typedef enum 
{
    AI_CB_TYPE_INVALID              = 0,        /* 无效枚举值 */
    AI_CB_TYPE_PROCESS_DONE         = 1,        /* 处理结束通知函数回调，函数指针对应 ai_process_done_func */
    AI_CB_TYPE_RELEASE_INPUT        = 2,        /* 算法释放调用者资源回调，函数指针对应 ai_release_func */
}ai_cb_type_e;

/************************************* 结构体定义 *************************************/

/* 方案版本号 */
typedef struct
{
    uint32_t protocol_version;              /* 协议文档版本 */

    char version[AI_VERSION_LEN];           /* 算法库版本 */
}ai_verion_s;

/* 算法能力 */
typedef struct
{
    /* 算法运行相关极值要求 */
    uint32_t    max_result_obj;                /* 最大输出目标数 */
    
    uint16_t    max_channel_nr;             /* 最大通道数 */
    uint16_t    max_batch_size;             /* 最大批处理数量 */
    
    uint16_t    max_rule_nr;                /* 最大规则数 */
    uint16_t    max_alert_nr;               /* 最大报警数 */
    
    uint32_t    max_image_width;            /* 最大图像宽度 */
    uint32_t    max_image_height;           /* 最大图像高度 */
    
    uint32_t    schedule_mode;              /* 算法库支持的主处理接口调用模式，可以是异步和同步的一种或两种 */

    uint32_t    reserved[26];
}ai_capacity_s;
AI_SIZEOF_CHECK_EQ(ai_capacity_s, 128);

/* 创建参数 */
typedef struct
{
    /* 算法参数 */
    uint32_t    result_obj;                 /* 调用者需要的目标数 */
    
    uint16_t    channel_nr;                 /* 调用者需要的最大通道数 */
    uint16_t    batch_size;                 /* 调用者需要的最大批处理数量 */
    
    uint16_t    rule_nr;                    /* 调用者需要的最大规则数 */
    uint16_t    alert_nr;                   /* 调用者需要的最大报警数 */

    uint32_t    coordinate;                 /* 输出目标矩形坐标系宽高，为0表示按实际图像宽高坐标输出； */
    uint32_t    schedule_mode;              /* 调度模式，对应枚举 ai_schedule_mode_e，只允许设置同步或异步的一种 */

    int32_t     chip_id;                    /* 多卡设备或多异构核芯片，指定算法运行的卡号或硬核ID */
    
    const char  *config_file;               /* 算法库配置文件路径，具体到文件名 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(config_file)

    uint32_t    reserved[24];
}ai_create_s;
AI_SIZEOF_CHECK_EQ(ai_create_s, 128);


/* 算法输入图像相关信息 */
typedef struct
{
    uint32_t    channel_id;                 /* 通道号 */
    uint32_t    frame_id;                   /* 帧序号 */

    uint32_t    width;                      /* 图像宽，单位为 像素点 */
    uint32_t    height;                     /* 图像高，单位为 像素点 */
    uint32_t    stride;                     /* 图像跨距，单位为 像素点 */

    uint8_t     colorspace;                 /* 图像颜色空间, 取值见枚举 ai_colorspace_e */
    uint8_t     light;                      /* 图像成像光源类型, 取值见枚举 ai_image_light_e */

    uint16_t    fps;                        /* 视频帧率,单位为帧每秒，仅处理视频流时有效 */

    /*  地址图像主数据为使用操作系统内存管理接口申请的用于存储图像数据的内存地址(以后简称OS内存)，
        当设备SDK提供内存管理接口支持申请包含OS内存和媒体内存(用于设备硬件外设计算的)的接口，
        那么主数据地址填OS内存地址，辅地址填媒体内存地址。
        对于主从应用，如英伟达卡，主内存为OS内存，辅内存为cuda内存。
        主内存必填，辅内存非必填。内存地址内容必须连续，且需按照算法需求做好颜色空间转换及地址对齐。
     */
    void        *main_data;                 /* 图像主数据地址，内存由调用者申请，在处理结束前，不得释放 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(main_data)
    
    void        *sub_data;                  /* 图像辅数据地址，内存由调用者申请，在处理结束前，不得释放 */  
    AI_UNUSED_FORCE_POINTER_ALIGN8(sub_data)

    uint32_t    reserved[22];
}ai_input_image_s;
AI_SIZEOF_CHECK_EQ(ai_input_image_s, 128);

/* 输入数据信息 */
typedef struct
{
    uint32_t    reserved[32];
}ai_input_info_s;
AI_SIZEOF_CHECK_EQ(ai_input_info_s, 128);

typedef struct
{
    uint32_t    frame_type;                 /* 输入数据类型，定义见枚举值 ai_input_type_e */
    void        *frame;                     /* 输入数据，根据frame_type确认其结构体 */
}ai_input_frame_s;

/* 算法处理输入信息，参照视频编码格式，输入信息可以只包含视频信息，也可以附加音频，数据等信息 */
typedef struct
{
    uint32_t            num;                /* 一包中数据的数量 */
    ai_input_frame_s    *frames;            /* 数据，使用时不可有相同类型的数据 */
}ai_input_s;

/* 矩形定义，以图像左上角为原点，向右向下为第一象限的直角坐标系 */
typedef struct
{
    /* 坐标框左上角坐标，lt:left-top， 注意根据坐标原点不同，坐标可能为负值 */
    int32_t     lt_x;                       /* 左上角x轴坐标 */
    int32_t     lt_y;                       /* 左上角y轴坐标 */

    uint32_t     width;                     /* 坐标框宽 */
    uint32_t     height;                    /* 坐标框高 */
}ai_rect_s;
AI_SIZEOF_CHECK_EQ(ai_rect_s, 16);

/* 单个目标基本信息 */
typedef struct
{
    uint16_t    id;                         /* 目标ID，本头文件中表示识别出来的目标的索引，仅用于区分一次识别中的不同目标 */
    uint16_t    type;                       /* 目标类别，具体含义由协议文档约束 */
    uint16_t    sub_type;                   /* 目标子类别，具体含义由协议文档根据主类别约束 */    

    uint16_t    confidence;                 /* 目标检测置信度，范围为 大小0 - 100 */
    uint16_t    state;                      /* 跟踪状态，定义见枚举 ai_obj_state_e */
    uint16_t    motion_status;              /* 运动状态，定义见枚举 ai_obj_motion_status_e */

    ai_rect_s   rect;                       /* 目标框，坐标系由创建参数中的 coordinate 字段确定 */
    
    uint32_t    reserved[9];
}ai_obj_base_s;
AI_SIZEOF_CHECK_EQ(ai_obj_base_s, 64);

/* 识别属性结果 */
typedef struct
{
    uint32_t    confidence;                 /* 属性置信度，范围为 大小0 - 100 */
    uint32_t    length;
    char        *prop_value;                /* 属性结果，json格式，utf8编码格式，其结构由协议文档约束 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(prop_value);

    uint32_t    reserved[12];
}ai_obj_property_s;
AI_SIZEOF_CHECK_EQ(ai_obj_property_s, 64);

/* 算法输出的单个目标信息 */
typedef struct
{
    ai_obj_base_s       base;               /* 目标基本信息 */

    /* 以下为识别结果定义 */
    ai_obj_property_s   *props;             /* 属性结果信息 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(props);

    uint32_t            num;               /* 属性结果数量 */
    
    uint32_t            reserved[13];
}ai_obj_s;
AI_SIZEOF_CHECK_EQ(ai_obj_s, 128);

/* 事件结果 */
typedef struct
{
    uint32_t        rule_id;                /* 报警事件对应的规则ID */
    
    uint32_t        obj_num;                /* 报警事件关联的目标数量 */
    ai_obj_base_s   objs;                   /* 报警事件关联的目标 */
    
    char            *event;                 /* 事件信息，json格式，utf8编码格式，其结构由协议文档约束 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(event)
    uint32_t        event_length;           /* 事件信息字符串长度 */    
    
    uint32_t        reserved[10];
}ai_event_s;
AI_SIZEOF_CHECK_EQ(ai_event_s, 128);

/* 算法结果 */
typedef struct
{
    uint32_t    channel_id;             /* 通道号 */
    uint32_t    frame_id;               /* 帧序号 */
    
    const char  *result;                /* json格式的结果输出，如果该字段为空则结果从下面结构体中输出；json格式定义见协议文档 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(result)

    uint32_t    result_length;          /* 事件数量 */
    uint32_t    reserved1;              /* 字节对齐保留位 */

    uint32_t    obj_num;                /* 目标数量 */
    uint32_t    event_num;              /* 事件数量 */

    ai_obj_s    *objs;                  /* 目标信息 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(objs)
    
    ai_event_s  *events;                /* 事件信息 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(events)
    
    uint32_t    reserved[20];
}ai_result_s;
AI_SIZEOF_CHECK_EQ(ai_result_s, 128);

/* 配置结构体 */
typedef struct
{
    int32_t     channel_id;             /* 需要配置的通道号，如果等于-1，则为广播 */
    uint32_t    type;                   /* 配置类型，定义见枚举 ai_config_type_e */
    
    void        *param;                 /* 配置数据 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(param)
    
    uint32_t    reserved[12];
}ai_config_s;
AI_SIZEOF_CHECK_EQ(ai_config_s, 64);

/* 规则配置 */
typedef struct
{
    uint32_t    rule_id;                /* 规则ID，用于区分规则的唯一标识 */
    
    uint32_t    rule_length;            /* 事件信息字符串长度 */
    const char  *rule;                  /* 事件信息，json格式，utf8编码格式，其结构由协议文档约束 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(rule)
        
    uint32_t    reserved[12];
}ai_config_rule_s;
AI_SIZEOF_CHECK_EQ(ai_config_rule_s, 64);

/* 规则配置 */
typedef struct
{
    ai_config_rule_s    *rules;         /* 规则信息 */
    AI_UNUSED_FORCE_POINTER_ALIGN8(rules)
    
    uint32_t            num;            /* 规则数 */

    uint32_t            reserved[13];
}ai_config_rules_s;
AI_SIZEOF_CHECK_EQ(ai_config_rules_s, 64);

/* 通道配置 */
typedef struct
{
    uint32_t    type;                   /* 通道配置类型，定义见枚举 ai_config_chennel_type_e */

    uint32_t    reserved[15];
}ai_config_channel_s;
AI_SIZEOF_CHECK_EQ(ai_config_channel_s, 64);

/* 异步调度模式下，需要注册的回调函数定义 */
typedef int32_t (*ai_release_func)(void *pCbParam, int32_t num, ai_input_s *pInput[] );
typedef int32_t (*ai_process_done_func)(void *pCbParam);

/***************************************** 接口定义 *****************************************/

/**-----------------------------------------------------------------------------------
\brief    算法初始化。
\note
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_init();

/**-----------------------------------------------------------------------------------
\brief    获取算法版本。
\note
\param    pVersion  获取的版本信息，内存由调用者申请
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_get_version( ai_verion_s *pVersion );

/**-----------------------------------------------------------------------------------
\brief    获取算法能力。
\note
\param    pCaps  算法能力参数，内存由调用者申请
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_get_capacities( ai_capacity_s *pCaps );

/**-----------------------------------------------------------------------------------
\brief    算法句柄创建。
\note
\param    ppHandle  返回给调用者的算法句柄
\param    pCreate   算法句柄创建参数
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_create( void **ppHandle, ai_create_s *pCreate );

/**-----------------------------------------------------------------------------------
\brief    异步调度模式下，注册处理结束回调用于通知调用者处理结束，注册结果释放回调函数。
\note
\param    pHandle  算法句柄
\param    cb_type   要注册的回调类型，定义见枚举 ai_cb_type_e
\param    cb_func   注册的回调函数指针
\param    cb_param  调用者私有参数，在算法库调用回调函数时传给回调函数
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_register_callback( void *pHandle, uint32_t cbType, void *pCbFunc, void *pCbParam);

/**-----------------------------------------------------------------------------------
\brief    异步调度模式下，调用者获取结果。
\note
\param    pHandle   算法句柄
\param    num       获取的结果数量
\param    pResult   获取的结果
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_get_result( void *pHandle, uint32_t *num, ai_result_s *pResult[] );

/**-----------------------------------------------------------------------------------
\brief    异步调度模式下，调用者释放结果，获取的结果有可能是算法内部申请的内存，所以需要告知算法结果可以释放。
\note
\param    pHandle   算法句柄
\param    num       要释放的结果数量
\param    pResult   要释放的结果
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_release_result( void *pHandle, uint32_t num, ai_result_s *pResult[] );

/**-----------------------------------------------------------------------------------
\brief    算法库主处理接口。
\note
\param    pHandle  算法句柄
\param    pInput   输入数据
\param    num      输入数据的数量
\param    pResult  同步模式下，返回的结果值
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_process( void *pHandle, ai_input_s *pInput, uint32_t num, ai_result_s *pSyncResult[]);

/**-----------------------------------------------------------------------------------
\brief    配置接口
\note
\param    pHandle   算法句柄
\param    pConfig   配置参数
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_config( void *pHandle, const ai_config_s *pConfig );

/**-----------------------------------------------------------------------------------
\brief    算法句柄销毁
\note
\param    pHandle  算法句柄
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_destory( void *pHandle );

/**-----------------------------------------------------------------------------------
\brief    算法反初始化。
\note
\return   错误码
--------------------------------------------------------------------------------------*/
int32_t ai_deinit();

#ifdef __cplusplus__
}
#endif

#endif