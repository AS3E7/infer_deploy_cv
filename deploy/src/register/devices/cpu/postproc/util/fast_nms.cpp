// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// typedef struct {
//     float x1, y1, x2, y2, score;
// } BBox;

// // 定义哈希函数
// int hash(float x1, float y1, float x2, float y2, int dim_x, int dim_y, int num_buckets) {
//     int x_bucket = (int)(x1 * dim_x);
//     int y_bucket = (int)(y1 * dim_y);
//     int w_bucket = (int)(x2 * dim_x) - x_bucket;
//     int h_bucket = (int)(y2 * dim_y) - y_bucket;
//     return (y_bucket + x_bucket * dim_y + w_bucket * dim_y * dim_x + h_bucket * dim_y * dim_x * num_buckets) % num_buckets;
// }

// // 定义桶结构体
// typedef struct {
//     int size;
//     BBox *boxes;
// } Bucket;

// // 创建桶
// void create_buckets(BBox *boxes, int num_boxes, int dim_x, int dim_y, int num_buckets, Bucket *buckets) {
//     for (int i = 0; i < num_buckets; i++) {
//         buckets[i].size = 0;
//         buckets[i].boxes = NULL;
//     }
//     for (int i = 0; i < num_boxes; i++) {
//         BBox *box = &boxes[i];
//         int idx = hash(box->x1, box->y1, box->x2, box->y2, dim_x, dim_y, num_buckets);
//         Bucket *bucket = &buckets[idx];
//         bucket->size++;
//         bucket->boxes = (BBox*)realloc(bucket->boxes, bucket->size * sizeof(BBox));
//         bucket->boxes[bucket->size - 1] = *box;
//     }
// }

// // 比较函数，用于按得分从高到低排序
// int compare(const void *a, const void *b) {
//     BBox *box1 = (BBox*)a;
//     BBox *box2 = (BBox*)b;
//     if (box1->score < box2->score) {
//         return 1;
//     }
//     else if (box1->score > box2->score) {
//         return -1;
//     }
//     else {
//         return 0;
//     }
// }

// // 计算IoU值
// #include <arm_neon.h>
// float iou(BBox *box1, BBox *box2) {
//     float32x4_t rect1_x1_y1_x2_y2, rect2_x1_y1_x2_y2;
//     float32x4_t left = vmaxq_f32(rect1_x1_y1_x2_y2, rect2_x1_y1_x2_y2);
//     float32x4_t rect1_width_height = vsubq_f32(rect1_x1_y1_x2_y2, left);
//     float32x4_t rect2_width_height = vsubq_f32(rect2_x1_y1_x2_y2, left);
//     float32x4_t zero = vdupq_n_f32(0);
//     float32x4_t w = vmaxq_f32(zero, vminq_f32(rect1_width_height, rect2_width_height));
//     float32x4_t h = vmaxq_f32(zero, vminq_f32(vsubq_f32(rect1_width_height, w), vsubq_f32(rect2_width_height, w)));
//     float32x4_t inter_area = vmulq_f32(w, h);
//     float32x4_t rect1_area = vmulq_f32(vsubq_f32(rect1_x1_y1_x2_y2, rect1_x1_y1_x2_y2), vsubq_f32(rect1_width_height, rect1_width_height));
//     float32x4_t rect2_area = vmulq_f32(vsubq_f32(rect2_x1_y1_x2_y2, rect2_x1_y1_x2_y2), vsubq_f32(rect2_width_height, rect2_width_height));
//     float32x4_t union_area = vsubq_f32(vaddq_f32(rect1_area, rect2_area), inter_area);
//     float32x4_t iou = vdivq_f32(inter_area, union_area);

//     float x1 = fmaxf(box1->x1, box2->x1);
//     float y1 = fmaxf(box1->y1, box2->y1);
//     float x2 = fminf(box1->x2, box2->x2);
//     float y2 = fminf(box1->y2, box2->y2);
//     float w = fmaxf(0.0f, x2 - x1 + 1.0f);
//     float h = fmaxf(0.0f, y2 - y1 + 1.0f);
//     float inter = w * h;
//     float area1 = (box1->x2 - box1->x1 + 1.0f) * (box1->y2 - box1->y1 + 1.0f);
//     float area2 = (box2->x2 - box2->x1 + 1.0f) * (box2->y2 - box2->y1 + 1.0f);
//     float iou = inter / (area1 + area2 - inter);
//     return iou;
// }

// // 进行NMS
// void nms_hashed(BBox *boxes, int num_boxes, float iou_threshold, int dim_x, int dim_y, int num_buckets) {
//     // 创建桶
//     Bucket *buckets = (Bucket*)malloc(num_buckets * sizeof(Bucket));
//     create_buckets(boxes, num_boxes, dim_x, dim_y, num_buckets, buckets);
//     // 遍历桶，对每个桶内的框进行去重
//     int offset = 0;
//     for (int i = 0; i < num_buckets; i++) {
//         Bucket *bucket = &buckets[i];
//         if (bucket->size <= 0) {
//             continue;
//         }
//         // 将框按得分从高到低排序
//         qsort(bucket->boxes, bucket->size, sizeof(BBox), compare);
//         // 逐个处理框
//         for (int j = 0; j < bucket->size; j++) {
//             BBox *box1 = &bucket->boxes[j];
//             if (box1->score <= 0.0f) {
//                 continue; // 跳过已经被删除的框
//             }
//             offset++;
//             // 对当前框和之后的框进行比较
//             for (int k = j + 1; k < bucket->size; k++) {
//                 BBox *box2 = &bucket->boxes[k];
//                 if (iou(box1, box2) > iou_threshold) {
//                     box2->score = 0.0f; // 删除与当前框重叠度高的框
//                 }
//             }
//         }
//     }
//     // 释放内存
//     for (int i = 0; i < num_buckets; i++) {
//         free(buckets[i].boxes);
//     }
//     free(buckets);
// }

// int main() {
//     int num_boxes = 5;
//     BBox boxes[] = {
//         {0.1f, 0.2f, 0.3f, 0.4f, 0.9f},
//         {0.2f, 0.3f, 0.4f, 0.5f, 0.8f},
//         {0.3f, 0.4f, 0.5f, 0.6f, 0.7f},
//         {0.4f, 0.5f, 0.6f, 0.7f, 0.6f},
//         {0.5f, 0.6f, 0.7f, 0.8f, 0.5f}
//     };
//     float iou_threshold = 0.5f;
//     int dim_x = 10;
//     int dim_y = 10;
//     int num_buckets = 100;
//     nms_hashed(boxes, num_boxes, iou_threshold, dim_x, dim_y, num_buckets);
//     for (int i = 0; i < num_boxes; i++) {
//         printf("%f %f %f %f %f\n", boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2, boxes[i].score);
//     }
//     return 0;
// }