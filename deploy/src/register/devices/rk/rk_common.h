#pragma once

#include "core/mem/buf_surface.h"

int convertSurfFormat2RKFormat(BufSurfaceColorFormat &format);
BufSurfaceColorFormat convertRKFormat2SurfFormat(int &format);
