---
layout: post
title: "Cross compiling ffmpeg for beaglebone"
description: ""
category: 
tags: [beaglebone ffmpeg crosscompile]
---
{% include JB/setup %}

# Introduction
This tutorial is mostly sourced from
[this](https://trac.ffmpeg.org/wiki/CompilationGuide/RaspberryPi) article on
building ffmpeg for RaspberryPi and also
[this](http://lakm.us/logit/2012/11/cross-compiling-ubuntu-arm-ffmpeg/) one showing
how to use checkinstall. While derek molloy has an excellent
[article](http://derekmolloy.ie/building-ffmpeg-for-beaglebone-from-source/) on
compiling ffmpeg from source, what he sets out there is compiling ffmpeg on the
beaglebone itself which, besides being really slow, is not possible on my older
beaglebone white with its lesser 256MB of RAM. The OS would always end up
killing the build process after some time throwing an 'Out of Memory Exception'.

# Set up a cross compilation environment

The first step will be to set up a build environment for the beaglebone which
means to set up a toolchain that targets the beaglebone hardware. I've posted a
detailed guide on this [here](2015-03-08-building-a-linaro-cross-compiler-toolchain-with-cross-ng.md).

# Download ffmpeg

For this guide, I used ffmpeg 2.6 which was the release at the time of
writing. You can either download the latest release from
[http://ffmpeg.org/download.html]() or clone from one of their git repository
mirrors listed on the bottom of the aforementioned page (such as
_git://source.ffmpeg.org/ffmpeg.git_).

# Compile

If you have any shared libraries that need to be linked to ffmpeg, install them
into a temp folder like `$HOME/arm-builds/ffmpeg`. For my purposes, I'll be
building ffmpeg with libaacplus. Now move to the source folder on the host
machine and configure the package.

    ./configure --enable-cross-compile --cross-prefix=${CCPREFIX} --arch=armhf \
    --target-os=linux --prefix=$HOME/arm-builds/ffmpeg/ --enable-nonfree \
    --enable-libaacplus --extra-cflags="-I$HOME/arm-builds/ffmpeg/include" \
    --extra-ldflags="-L$HOME/arm-builds/ffmpeg/lib" --extra-libs=-ldl \
    --enable-shared

If you don't need to build ffmpeg without libaacplus or any additional libraries you can scrap
the `--enable-libaacplus --extra-libs --extra-ldflags --extra-cflags --enable-nonfree`.

Now go ahead and do a make

    make

This will take some time. Once that's compiled we copy the whole stinkin' build
folder to the beaglebone by way of rsync,scp, or whatever you prefer. Once
that's copied, we will install the package using checkinstall. Checkinstall is
clutch because it actually builds a debian package of the compiled code and then
installs that which makes it easy to uninstall later (it can even be
redistributed this way). Move into the copied source folder and run the
following command

    sudo checkinstall --pkgname=ffmpeg --pkgversion="5:$(./version.sh)" --backup=no --deldoc=yes --default

# That's it!

ffmpeg should be installed now and you're also left with a redistributable
ffmpeg.deb package.



