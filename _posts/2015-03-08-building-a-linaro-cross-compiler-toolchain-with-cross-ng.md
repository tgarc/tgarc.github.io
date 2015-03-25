---
layout: post
title: "Building a custom toolchain for the BeagleBone"
description: ""
category: 
tags: [linaro arm cross-compile beaglebone]
---
{% include JB/setup %}

This post is about creating a custom (linaro) toolchain using using
cross-ng. The motivation for this came about when I found I was unable to build
the ffmpeg library directly on a beaglebone (specifically, the older beaglebone
white model) because of insufficient memory during the build process [1].

First, we need to download cross-ng with linaro support. I used the source at
[https://github.com/Christopher83/linaro_crosstool-ng](). To install crosstool-ng:

    git clone https://github.com/Christopher83/linaro_crosstool-ng
    cd linaro_crosstool-ng
    ./bootstrap
    ./configure --prefix=/install/path (e.g. /opt/cross)
    make
    sudo make install
    
Then, make a directory to build your tool chain

    mkdir /home/<user>/ctng
    cd /home/<user>/ctng

Now, we configure the toolchain using the crosstool-ng interface

    /opt/cross/bin/ct-ng menuconfig

* Under `Path and misc options`
  * Select `Try features marked as EXPERIMENTAL`
  * Set `Prefix directory` to `${HOME}/x-tools/${CT_TARGET}' or whatever you choose [2]
  * Set `Number of parallel jobs` to *2* times the number of CPU cores on the host
* Under `Operating Sytems` change `Target OS` to `linux`
* Under `Target options`
  * Set `Target Architecture` to `arm`
  * Select `Use the MMU`
  * Set `Endianness` to `Little endian`
  * Set `Default instruction set mode` to `thumb`
  * Select `Use EABI`
  * Set `Tune for CPU` to `cortex-a8`
  * Set `Floating point` to `hardware (FPU)`
  * Set `Use specific FPU` to `neon`
  * Set `Architecture level` to `armv7-a`
* Under `Toolchain Options` set `Tuple's Vendor String` to `linaro`
* Under `C Library`
  * Set `C Library` to `eglibc`
  * (Important) Set `eglibc version` to `Linaro 2.19-2014.08` (should be same as version of glibc on target)
* Under `C Compiler`
  * Select `Show Linaro versions`
  * Set gcc version to latest linaro-gcc
  * Optionally, select additional supported languages (e.g. C++)

Exit and save settings. Then, to build the toolchain, type

    /opt/cross/bin/ct-ng build

This will take a while (20 min on my quad core i7). When it's done create a file
containing the following lines:

    export PATH=$PATH:$HOME/x-tools/arm-linaro-linux-gnueabihf/bin
    export CCPREFIX="$HOME/x-tools/arm-linaro-linux-gnueabihf/bin/arm-linaro-linux-gnueabihf-"

You can now source this file anytime to set up the environment variables
necessary to compile with your toolchain. Let's sanity check the compiler:

    $ arm-linaro-linux-gnueabihf-gcc --version
    arm-linaro-linux-gnueabihf-gcc (crosstool-NG 1.20.0) 4.9.3 20150209 (prerelease)
    Copyright (C) 2014 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    $ cat > test.c
    #include <stdio.h>
    int main() { printf("Hello, world!\n"); return 0; }
    ^D

    $ arm-linaro-linux-gnueabi-gcc -o test test.c

Copy the test executable from this last step to your beaglebone and execute it
there. You should see the Hello World output. Success!


[1]: It is still possible to build directly on the beaglebone by
adding swap space.

[2]: I settled for this instead of /opt/cross/x-tools since the crosstool build
cannot be run as root but writing to /opt/cross requires root.
