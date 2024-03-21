#!/bin/bash
# Synchronize the notebooks and output directories with the remote on a vast instance
# First syncs the local data up to the remote, then loops on syncing
# the remote to local.
# The initial sync of local to remote can be skipped with -r (for remote only)
# This mode is useful if the sync loop is interrupted and you need to restart it.
# The ssh command provided on the command line must be in the format provided by
# vast.ai when the "Connect_>" button on an instance card is pushed
set -x

usage() {
    echo "Usage: $0 [-r] [-s sync_frequency] vast_connect_ssh_command"  1>&2
    exit 1
}


SYNC_FREQUENCY=10 # default sync frequency is 10 seconds
# Process command-line options.  For now, just -r
while [ "${1:0:1}" == '-' ]; do
    option="$1"
    shift
    if [ "$option" == '-r' ] ; then
        SKIP_PUSH=1
        echo skipping push
    elif [ "$option" == '-s' ] ; then
        SYNC_FREQUENCY="$1"
        # check that SYNC_FREQUENCY looks like an integer
        if ! [[ "$SYNC_FREQUENCY" =~ ^[0-9]+$ ]] ; then
            echo "SYNC_FREQUENCY must be an integer" 1>&2
            exit 1
        fi
        shift
    else
        echo "unrecognized option: $option" 1>&2
        usage
        exit 1
    fi
done

# rsync synchronize from remote to local every SYNC_FREQUENCY seconds

# Check if the argument is provided and matches /^ssh /'
if [ "$#" -ne 1 -o "${1:0:4}" != 'ssh ' ]; then
    usage
fi

SSH_COMMAND="$1"

# Extract user, host, and port from the SSH command
export HOST=$(echo "$SSH_COMMAND" | perl -ne 'print $1 if /@(\S+) -L/')
export USER=$(echo "$SSH_COMMAND" | perl -ne 'print $1 if /ssh -p \d+ (\S+)@/')
export PORT=$(echo "$SSH_COMMAND" | perl -ne 'print $1 if /-p (\d+)/')

echo HOST=$HOST
echo PORT=$PORT
echo USER=$USER

# if any of HOST, PORT, USER are empty strings, complain about the format of the ssh command and exit with error
if [ -z "$HOST" -o -z "$PORT" -o -z "$USER" ]; then
    echo "Format of ssh command must be: ssh -p <port> <user>@<host> -L <local_port>:<host>:<remote_port>"
    exit 1
fi


LOCAL_OUTPUTS_DIR='./outputs'
REMOTE_OUTPUTS_DIR='/root/outputs'
LOCAL_NOTEBOOKS_DIR='./notebooks'
REMOTE_NOTEBOOKS_DIR='/root/notebooks'
EXCLUDE_ARGS="--exclude .git/ --exclude memo_cache/ --exclude __pycache__/ --exclude .*.swp"

# if the SKIP_PUSH varaible is not defined
if [ -z "$SKIP_PUSH" ]; then
# Initial sync from local to remote
    echo pushing first
    [ -d "$LOCAL_NOTEBOOKS_DIR" ] || { echo "$LOCAL_NOTEBOOKS_DIR" directory does not exist 1>&2 ; exit 1 ; }
    rsync -avz $EXCLUDE_ARGS -e "ssh -p $PORT" "$LOCAL_NOTEBOOKS_DIR/" "$USER@$HOST:$REMOTE_NOTEBOOKS_DIR"

    [ -d "$LOCAL_OUTPUTS_DIR" ] || { echo "$LOCAL_OUTPUTS_DIR" directory does not exist 1>&2 ; exit 1 ; }
    rsync -avz $EXCLUDE_ARGS -e "ssh -p $PORT" "$LOCAL_OUTPUTS_DIR/" "$USER@$HOST:$REMOTE_OUTPUTS_DIR"
else
    echo skipping push
fi

# Function to sync from remote to local
sync_from_remote() {
    remote_dir="$1"
    local_dir="$2"
    rsync_output=$(rsync -avz $EXCLUDE_ARGS -e "ssh -p $PORT" "$USER@$HOST:$remote_dir/" "$local_dir" 2>&1)
    if [ $? -ne 0 ]; then
        echo "rsync problem for remote:$remote_dir to local:$local_dir: $rsync_output" 1>&2
    fi
}

set +x

# Main loop for periodic sync
while true; do
    sync_from_remote "$REMOTE_OUTPUTS_DIR" "$LOCAL_OUTPUTS_DIR"
    sync_from_remote "$REMOTE_NOTEBOOKS_DIR" "$LOCAL_NOTEBOOKS_DIR"
    sleep $SYNC_FREQUENCY # Sync every SYNC_FREQUENCY seconds
done
