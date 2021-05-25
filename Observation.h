#pragma once
#include "NeuralNetwork.h"
#include "TensorObserver.h"

void ObserveSingleStepTest(const ConvolutionLayerOb& ob);
void ObserveSingleStepConv1(const ConvolutionLayerOb& ob);
void ObserveSingleStepConv2_1(const ConvolutionLayerOb& ob);
void ObserveSingleStepConv2_2(const ConvolutionLayerOb& ob);
void ObserveSingleStepMaxInputSum(const ConvolutionLayerOb& ob);

void ObserveSerialStepsNegativeLearningSuppress(const ConvolutionLayerOb& ob);
void ObserveSerialStepsNewFeaturesOnInput(const ConvolutionLayerOb& ob);

void ObserveSerialStepsTargetFeature(const ConvolutionLayerOb& ob, const int i_feature);
void ObserveSerialStepsTargetFeatureActive(const ConvolutionLayerOb& ob, const int i_feature);
void ObserveSerialStepsTargetFeatureNegativeLearningSuppress(const ConvolutionLayerOb& ob, const int i_feature);
void ObserveSerialStepsTargetFeatureSuppressed(const ConvolutionLayerOb& ob, const int i_feature);

void Observe(const int i_layer, const bool is_learning, const ConvolutionLayerOb& ob);
