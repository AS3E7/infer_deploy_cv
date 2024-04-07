BASEDIR=$(pwd)
TASK_TYPE=$1
URL=$2
TASK_ID=$3
SUB_TASK_STATE=$4 

if [ ! -n $5 ]; then
    IS_QUANTIZATION=None
    python3 $BASEDIR/launch.py --task_type $TASK_TYPE  --url $URL --task_id $TASK_ID --sub_task_state $SUB_TASK_STATE 
else
    IS_QUANTIZATION=$5
    python3 $BASEDIR/launch.py --task_type $TASK_TYPE  --url $URL --task_id $TASK_ID --sub_task_state $SUB_TASK_STATE --is_quantization $IS_QUANTIZATION
fi
#python3 $BASEDIR/launch.py --task_type $TASK_TYPE  --url $URL --task_id $TASK_ID --sub_task_state $SUB_TASK_STATE --is_quantization $IS_QUANTIZATION
