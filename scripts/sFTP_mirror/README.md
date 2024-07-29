# Mirroring sFTP server with models

To organize a local copy and a global sFTP, do the following:
1. Organize local sFTP server.
2. Copy cache from global sFTP.
3. Setup periodical mirrroring of the global cache.
4. Create configuration file

Default configurations stores in file `./scripts/sFTP_mirror/common.sh`.

You can set the following parameters:

1. name of default sFTP user with read-only access (`sftp_user`),
2. path to the user's home directory (`user_home_dir`),
3. default group name for sFTP users (`group`).


## Organizing local sFTP server


### Automatic script

```
cd ./scripts/sFTP_mirror
sudo ./setup_sft_server.sh
```

You can find description of actions below.

### Manual actions

1. Go to directory with scripts: `cd ./scripts/sFTP_mirror`.
2. Load parameters from `common.sh`: `source ./common.sh`.
3. Add sFTP group by command: `addgroup sftp`.
4. Comment list with string `Subsystem` in file `/etc/ssh/sshd_config`.
5. Deny access by SSH to all users in `sftp` group and restrct access by sFTP to file system for them out of home directory. Add the following commands to file `/etc/ssh/sshd_config`:
```
Subsystem    sftp    internal-sftp
Match group sftp
ForceCommand internal-sftp
PasswordAuthentication yes
ChrootDirectory %h
PermitTunnel no
AllowAgentForwarding no
AllowTcpForwarding no
X11Forwarding no
```
6. Create user which will be used for read-only access to the data:
```
useradd ${sftp_user} -d ${user_home_dir} -g ${group} -m
```
7. Set user's password:
```
passwd ${sftp_user}
```

8. Set owner of the directory:
```
chown root:root ${user_home_dir}
chown ${sftp_user}:${group} ${user_home_dir}/cache
```

9. Restart SSHD service: `sudo service sshd restart`

## Copy cache from global sFTP

### Automatic script

```
cd ./scripts/sFTP_mirror
source ./common.sh
./downlaod_cache.sh ${user_home_dir}
```

You can find description of actions below.

### Manual actions

1. Go to directory with scripts: `cd ./scripts/sFTP_mirror`.
2. Load parameters from `common.sh`: `source ./common.sh`.
3. Run command: `wget --mirror -pc --convert-links --reject "index.html*" -nH --no-parent -e robots=off -k -P  ${user_home_dir} 'https://jpeg-git.lx.it.pt/cache/'`


## Setup periodical mirrroring of the global cache

### Automatic script

```
sudo ./scripts/sFTP_mirror/setup_crontab.sh
```

You can find description of actions below.

### Manual actions

1. Go to directory with scripts: `cd ./scripts/sFTP_mirror`.
2. Load parameters from `common.sh`: `source ./common.sh`.
3. Store current dir to variable : ``SCRIPT_DIR=`pwd` ``
4. Add the following lines to `/etc/crontab`: `echo -e "0 */3\t* * *\t${sftp_user}\t${SCRIPT_DIR}/cron.sh" >> /etc/crontab`

## Create configuration file of DVC for users

1. Run script `./scripts/sFTP_mirror/setup_cfg.sh`
2. Enter IP of sFTP server.
3. Enter sFTP's user password.
4. Configuration file will be stored to `.dvc/config.local`.

You can share file `.dvc/config.local` to other users if you would like to give them read/write access on local sFTP with models' storage.
