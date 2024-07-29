#!/bin/bash

source ./common.sh

cfg=/etc/ssh/sshd_config

addgroup ${group}
sed -i 's/Subsystem/#Subsystem/' ${cfg}
tee -a ${cfg} > /dev/null <<EOT
Subsystem    sftp    internal-sftp
Match group ${group}
ForceCommand internal-sftp
PasswordAuthentication yes
ChrootDirectory %h
PermitTunnel no
AllowAgentForwarding no
AllowTcpForwarding no
X11Forwarding no
EOT


useradd ${sftp_user} -d ${user_home_dir} -g ${group} -m
passwd ${sftp_user}

mkdir ${user_home_dir}/cache
chown root:root ${user_home_dir}
chown ${sftp_user}:${group} ${user_home_dir}/cache
chmod g+w ${user_home_dir}/cache

service sshd restart
