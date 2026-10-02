#ifndef ZERO_EVALUATION_H
#define ZERO_EVALUATION_H

#include "position.h"
#include "evaluation/material.h"
#include "evaluation/pawns.h"
#include "evaluation/mobility.h"
#include "evaluation/king.h"

int evaluatePosition(const Position& position);

#endif
