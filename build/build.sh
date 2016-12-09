#! /bin/sh -ex

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ../

pgrep rdm || nohup ~/.emacs.d/lisp/rtags/bin/rdm &

until ~/.emacs.d/lisp/rtags/bin/rc -J .;
do
    sleep 1;
done

make -j`nproc`
