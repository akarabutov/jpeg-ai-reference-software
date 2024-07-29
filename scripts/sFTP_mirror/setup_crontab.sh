#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${SCRIPT_DIR}/common.sh

echo -e "0 */3\t* * *\t${sftp_user}\t${SCRIPT_DIR}/cron.sh" >> /etc/crontab
