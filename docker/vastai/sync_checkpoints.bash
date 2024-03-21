#!/usr/bin/env bash
# upload or download the contents of the $OUTPUT_DIR to the S3 bucket $CHECKPOINTS_BUCKET with prefix $CHECKPOINTS_PREFIX

# direction must be "up" or "down"
DIRECTION="$1"

if [ -z "$AWS_PROFILE" ]
then 
  if [ -z "$AWS_ACCESS_KEY_ID" -o -z "$AWS_SECRET_ACCESS_KEY" ]
  then
    echo "AWS_PROFILE or AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set" 1>&2
    exit 1
  fi
fi

if [ -z "$CHECKPOINTS_BUCKET" ]
then
    echo "CHECKPOINTS_BUCKET must be defined to sync checkpoints" 1>&2
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]
then
    echo "OUTPUT_DIR must be defined to sync checkpoints" 1>&2
    exit 1
fi

# use specified CHECKPOINTS_PREFIX or default to /checkpoints
CHECKPOINTS_PREFIX=${CHECKPOINTS_PREFIX:-/checkpoints}
# strip any trailing slashes from CHECKPOINTS_PREFIX
CHECKPOINTS_PREFIX=$(echo "$CHECKPOINTS_PREFIX" | sed -e 's:/*$::')

S3_URI="s3://${CHECKPOINTS_BUCKET}${CHECKPOINTS_PREFIX}"

set -x
case $DIRECTION in
  up)
    aws s3 sync "$OUTPUT_DIR" "$S3_URI" || exit $?
    ;;
  down)
    aws s3 sync "$S3_URI" "$OUTPUT_DIR" || exit $? 
    ;;
  *)
    set +x
    echo "DIRECTION must be either up or down" 1>&2
    exit 1
    ;;
esac
set +x

