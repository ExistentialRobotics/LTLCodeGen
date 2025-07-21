docker run --privileged --restart always -t \
    -v /var/run/docker.sock:/var/run/docker.sock:rw \
    -v /dev/char:/dev/char:rw \
    -v /dev/shm:/dev/shm:rw \
    -v /dev/bus:/dev/bus:rw \
    -v /dev/block:/dev/block:rw \
    -v /dev/serial:/dev/serial:rw \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${HOME}/.Xauthority:/root/.Xauthority:rw \
    --workdir /opt/ros_ws \
    -e DISPLAY \
    --net=host \
    --detach \
    --hostname container-ltlcodegen \
    --add-host=container-ltlcodegen:127.0.0.1 \
    --name ltlcodegen \
    --cap-add sys_ptrace --cpus=0 --memory-swap=-1 --ipc=host \
    ltlcodegen bash -l