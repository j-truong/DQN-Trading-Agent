ё	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
?
price_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameprice_layer1/kernel

'price_layer1/kernel/Read/ReadVariableOpReadVariableOpprice_layer1/kernel*"
_output_shapes
: *
dtype0
z
price_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameprice_layer1/bias
s
%price_layer1/bias/Read/ReadVariableOpReadVariableOpprice_layer1/bias*
_output_shapes
: *
dtype0
?
price_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameprice_layer2/kernel

'price_layer2/kernel/Read/ReadVariableOpReadVariableOpprice_layer2/kernel*"
_output_shapes
: *
dtype0
z
price_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameprice_layer2/bias
s
%price_layer2/bias/Read/ReadVariableOpReadVariableOpprice_layer2/bias*
_output_shapes
:*
dtype0
?
fixed_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*$
shared_namefixed_layer1/kernel
{
'fixed_layer1/kernel/Read/ReadVariableOpReadVariableOpfixed_layer1/kernel*
_output_shapes

:
*
dtype0
z
fixed_layer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefixed_layer1/bias
s
%fixed_layer1/bias/Read/ReadVariableOpReadVariableOpfixed_layer1/bias*
_output_shapes
:*
dtype0
?
fixed_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_namefixed_layer2/kernel
{
'fixed_layer2/kernel/Read/ReadVariableOpReadVariableOpfixed_layer2/kernel*
_output_shapes

:*
dtype0
z
fixed_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namefixed_layer2/bias
s
%fixed_layer2/bias/Read/ReadVariableOpReadVariableOpfixed_layer2/bias*
_output_shapes
:*
dtype0
?
action_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameaction_output/kernel
}
(action_output/kernel/Read/ReadVariableOpReadVariableOpaction_output/kernel*
_output_shapes

:*
dtype0
|
action_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameaction_output/bias
u
&action_output/bias/Read/ReadVariableOpReadVariableOpaction_output/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
?&
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?%
value?%B?% B?%
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
	optimizer
loss
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
 trainable_variables
!	keras_api
R
"	variables
#regularization_losses
$trainable_variables
%	keras_api
 
R
&	variables
'regularization_losses
(trainable_variables
)	keras_api
h

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
h

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
6
<iter
	=decay
>learning_rate
?momentum
 
F
0
1
2
3
*4
+5
06
17
68
79
 
F
0
1
2
3
*4
+5
06
17
68
79
?
@metrics
	variables
Anon_trainable_variables
Blayer_metrics

Clayers
Dlayer_regularization_losses
regularization_losses
trainable_variables
 
_]
VARIABLE_VALUEprice_layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Emetrics
	variables
Fnon_trainable_variables

Glayers
regularization_losses
Hlayer_regularization_losses
Ilayer_metrics
trainable_variables
_]
VARIABLE_VALUEprice_layer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Jmetrics
	variables
Knon_trainable_variables

Llayers
regularization_losses
Mlayer_regularization_losses
Nlayer_metrics
trainable_variables
 
 
 
?
Ometrics
	variables
Pnon_trainable_variables

Qlayers
regularization_losses
Rlayer_regularization_losses
Slayer_metrics
 trainable_variables
 
 
 
?
Tmetrics
"	variables
Unon_trainable_variables

Vlayers
#regularization_losses
Wlayer_regularization_losses
Xlayer_metrics
$trainable_variables
 
 
 
?
Ymetrics
&	variables
Znon_trainable_variables

[layers
'regularization_losses
\layer_regularization_losses
]layer_metrics
(trainable_variables
_]
VARIABLE_VALUEfixed_layer1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
?
^metrics
,	variables
_non_trainable_variables

`layers
-regularization_losses
alayer_regularization_losses
blayer_metrics
.trainable_variables
_]
VARIABLE_VALUEfixed_layer2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
?
cmetrics
2	variables
dnon_trainable_variables

elayers
3regularization_losses
flayer_regularization_losses
glayer_metrics
4trainable_variables
`^
VARIABLE_VALUEaction_output/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEaction_output/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71
 

60
71
?
hmetrics
8	variables
inon_trainable_variables

jlayers
9regularization_losses
klayer_regularization_losses
llayer_metrics
:trainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

m0
 
 
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ntotal
	ocount
p	variables
q	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

n0
o1

p	variables
|
serving_default_env_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_price_inputPlaceholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_env_inputserving_default_price_inputprice_layer1/kernelprice_layer1/biasprice_layer2/kernelprice_layer2/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_356053863
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'price_layer1/kernel/Read/ReadVariableOp%price_layer1/bias/Read/ReadVariableOp'price_layer2/kernel/Read/ReadVariableOp%price_layer2/bias/Read/ReadVariableOp'fixed_layer1/kernel/Read/ReadVariableOp%fixed_layer1/bias/Read/ReadVariableOp'fixed_layer2/kernel/Read/ReadVariableOp%fixed_layer2/bias/Read/ReadVariableOp(action_output/kernel/Read/ReadVariableOp&action_output/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_save_356054234
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameprice_layer1/kernelprice_layer1/biasprice_layer2/kernelprice_layer2/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference__traced_restore_356054292??
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_356054085

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?\
?	
$__inference__wrapped_model_356053481
price_input
	env_inputD
@model_2_price_layer1_conv1d_expanddims_1_readvariableop_resource8
4model_2_price_layer1_biasadd_readvariableop_resourceD
@model_2_price_layer2_conv1d_expanddims_1_readvariableop_resource8
4model_2_price_layer2_biasadd_readvariableop_resource7
3model_2_fixed_layer1_matmul_readvariableop_resource8
4model_2_fixed_layer1_biasadd_readvariableop_resource7
3model_2_fixed_layer2_matmul_readvariableop_resource8
4model_2_fixed_layer2_biasadd_readvariableop_resource8
4model_2_action_output_matmul_readvariableop_resource9
5model_2_action_output_biasadd_readvariableop_resource
identity??,model_2/action_output/BiasAdd/ReadVariableOp?+model_2/action_output/MatMul/ReadVariableOp?+model_2/fixed_layer1/BiasAdd/ReadVariableOp?*model_2/fixed_layer1/MatMul/ReadVariableOp?+model_2/fixed_layer2/BiasAdd/ReadVariableOp?*model_2/fixed_layer2/MatMul/ReadVariableOp?+model_2/price_layer1/BiasAdd/ReadVariableOp?7model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOp?+model_2/price_layer2/BiasAdd/ReadVariableOp?7model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOp?
*model_2/price_layer1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_2/price_layer1/conv1d/ExpandDims/dim?
&model_2/price_layer1/conv1d/ExpandDims
ExpandDimsprice_input3model_2/price_layer1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2(
&model_2/price_layer1/conv1d/ExpandDims?
7model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@model_2_price_layer1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype029
7model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOp?
,model_2/price_layer1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_2/price_layer1/conv1d/ExpandDims_1/dim?
(model_2/price_layer1/conv1d/ExpandDims_1
ExpandDims?model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOp:value:05model_2/price_layer1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2*
(model_2/price_layer1/conv1d/ExpandDims_1?
model_2/price_layer1/conv1dConv2D/model_2/price_layer1/conv1d/ExpandDims:output:01model_2/price_layer1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
model_2/price_layer1/conv1d?
#model_2/price_layer1/conv1d/SqueezeSqueeze$model_2/price_layer1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2%
#model_2/price_layer1/conv1d/Squeeze?
+model_2/price_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_2_price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_2/price_layer1/BiasAdd/ReadVariableOp?
model_2/price_layer1/BiasAddBiasAdd,model_2/price_layer1/conv1d/Squeeze:output:03model_2/price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
model_2/price_layer1/BiasAdd?
model_2/price_layer1/ReluRelu%model_2/price_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
model_2/price_layer1/Relu?
*model_2/price_layer2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*model_2/price_layer2/conv1d/ExpandDims/dim?
&model_2/price_layer2/conv1d/ExpandDims
ExpandDims'model_2/price_layer1/Relu:activations:03model_2/price_layer2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2(
&model_2/price_layer2/conv1d/ExpandDims?
7model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp@model_2_price_layer2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype029
7model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOp?
,model_2/price_layer2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_2/price_layer2/conv1d/ExpandDims_1/dim?
(model_2/price_layer2/conv1d/ExpandDims_1
ExpandDims?model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOp:value:05model_2/price_layer2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2*
(model_2/price_layer2/conv1d/ExpandDims_1?
model_2/price_layer2/conv1dConv2D/model_2/price_layer2/conv1d/ExpandDims:output:01model_2/price_layer2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_2/price_layer2/conv1d?
#model_2/price_layer2/conv1d/SqueezeSqueeze$model_2/price_layer2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2%
#model_2/price_layer2/conv1d/Squeeze?
+model_2/price_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_2_price_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_2/price_layer2/BiasAdd/ReadVariableOp?
model_2/price_layer2/BiasAddBiasAdd,model_2/price_layer2/conv1d/Squeeze:output:03model_2/price_layer2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model_2/price_layer2/BiasAdd?
model_2/price_layer2/ReluRelu%model_2/price_layer2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
model_2/price_layer2/Relu?
*model_2/average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*model_2/average_pooling1d_2/ExpandDims/dim?
&model_2/average_pooling1d_2/ExpandDims
ExpandDims'model_2/price_layer2/Relu:activations:03model_2/average_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2(
&model_2/average_pooling1d_2/ExpandDims?
#model_2/average_pooling1d_2/AvgPoolAvgPool/model_2/average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2%
#model_2/average_pooling1d_2/AvgPool?
#model_2/average_pooling1d_2/SqueezeSqueeze,model_2/average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2%
#model_2/average_pooling1d_2/Squeeze?
model_2/price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_2/price_flatten/Const?
model_2/price_flatten/ReshapeReshape,model_2/average_pooling1d_2/Squeeze:output:0$model_2/price_flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
model_2/price_flatten/Reshape?
 model_2/concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_2/concat_layer/concat/axis?
model_2/concat_layer/concatConcatV2&model_2/price_flatten/Reshape:output:0	env_input)model_2/concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????
2
model_2/concat_layer/concat?
*model_2/fixed_layer1/MatMul/ReadVariableOpReadVariableOp3model_2_fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02,
*model_2/fixed_layer1/MatMul/ReadVariableOp?
model_2/fixed_layer1/MatMulMatMul$model_2/concat_layer/concat:output:02model_2/fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/fixed_layer1/MatMul?
+model_2/fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_2_fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_2/fixed_layer1/BiasAdd/ReadVariableOp?
model_2/fixed_layer1/BiasAddBiasAdd%model_2/fixed_layer1/MatMul:product:03model_2/fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/fixed_layer1/BiasAdd?
model_2/fixed_layer1/ReluRelu%model_2/fixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/fixed_layer1/Relu?
*model_2/fixed_layer2/MatMul/ReadVariableOpReadVariableOp3model_2_fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*model_2/fixed_layer2/MatMul/ReadVariableOp?
model_2/fixed_layer2/MatMulMatMul'model_2/fixed_layer1/Relu:activations:02model_2/fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/fixed_layer2/MatMul?
+model_2/fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_2_fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_2/fixed_layer2/BiasAdd/ReadVariableOp?
model_2/fixed_layer2/BiasAddBiasAdd%model_2/fixed_layer2/MatMul:product:03model_2/fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/fixed_layer2/BiasAdd?
model_2/fixed_layer2/ReluRelu%model_2/fixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/fixed_layer2/Relu?
+model_2/action_output/MatMul/ReadVariableOpReadVariableOp4model_2_action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+model_2/action_output/MatMul/ReadVariableOp?
model_2/action_output/MatMulMatMul'model_2/fixed_layer2/Relu:activations:03model_2/action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/action_output/MatMul?
,model_2/action_output/BiasAdd/ReadVariableOpReadVariableOp5model_2_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_2/action_output/BiasAdd/ReadVariableOp?
model_2/action_output/BiasAddBiasAdd&model_2/action_output/MatMul:product:04model_2/action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/action_output/BiasAdd?
IdentityIdentity&model_2/action_output/BiasAdd:output:0-^model_2/action_output/BiasAdd/ReadVariableOp,^model_2/action_output/MatMul/ReadVariableOp,^model_2/fixed_layer1/BiasAdd/ReadVariableOp+^model_2/fixed_layer1/MatMul/ReadVariableOp,^model_2/fixed_layer2/BiasAdd/ReadVariableOp+^model_2/fixed_layer2/MatMul/ReadVariableOp,^model_2/price_layer1/BiasAdd/ReadVariableOp8^model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOp,^model_2/price_layer2/BiasAdd/ReadVariableOp8^model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2\
,model_2/action_output/BiasAdd/ReadVariableOp,model_2/action_output/BiasAdd/ReadVariableOp2Z
+model_2/action_output/MatMul/ReadVariableOp+model_2/action_output/MatMul/ReadVariableOp2Z
+model_2/fixed_layer1/BiasAdd/ReadVariableOp+model_2/fixed_layer1/BiasAdd/ReadVariableOp2X
*model_2/fixed_layer1/MatMul/ReadVariableOp*model_2/fixed_layer1/MatMul/ReadVariableOp2Z
+model_2/fixed_layer2/BiasAdd/ReadVariableOp+model_2/fixed_layer2/BiasAdd/ReadVariableOp2X
*model_2/fixed_layer2/MatMul/ReadVariableOp*model_2/fixed_layer2/MatMul/ReadVariableOp2Z
+model_2/price_layer1/BiasAdd/ReadVariableOp+model_2/price_layer1/BiasAdd/ReadVariableOp2r
7model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOp7model_2/price_layer1/conv1d/ExpandDims_1/ReadVariableOp2Z
+model_2/price_layer2/BiasAdd/ReadVariableOp+model_2/price_layer2/BiasAdd/ReadVariableOp2r
7model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOp7model_2/price_layer2/conv1d/ExpandDims_1/ReadVariableOp:X T
+
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?P
?
F__inference_model_2_layer_call_and_return_conditional_losses_356053977
inputs_0
inputs_1<
8price_layer1_conv1d_expanddims_1_readvariableop_resource0
,price_layer1_biasadd_readvariableop_resource<
8price_layer2_conv1d_expanddims_1_readvariableop_resource0
,price_layer2_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?#price_layer1/BiasAdd/ReadVariableOp?/price_layer1/conv1d/ExpandDims_1/ReadVariableOp?#price_layer2/BiasAdd/ReadVariableOp?/price_layer2/conv1d/ExpandDims_1/ReadVariableOp?
"price_layer1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"price_layer1/conv1d/ExpandDims/dim?
price_layer1/conv1d/ExpandDims
ExpandDimsinputs_0+price_layer1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
price_layer1/conv1d/ExpandDims?
/price_layer1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8price_layer1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/price_layer1/conv1d/ExpandDims_1/ReadVariableOp?
$price_layer1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$price_layer1/conv1d/ExpandDims_1/dim?
 price_layer1/conv1d/ExpandDims_1
ExpandDims7price_layer1/conv1d/ExpandDims_1/ReadVariableOp:value:0-price_layer1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2"
 price_layer1/conv1d/ExpandDims_1?
price_layer1/conv1dConv2D'price_layer1/conv1d/ExpandDims:output:0)price_layer1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
price_layer1/conv1d?
price_layer1/conv1d/SqueezeSqueezeprice_layer1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
price_layer1/conv1d/Squeeze?
#price_layer1/BiasAdd/ReadVariableOpReadVariableOp,price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#price_layer1/BiasAdd/ReadVariableOp?
price_layer1/BiasAddBiasAdd$price_layer1/conv1d/Squeeze:output:0+price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
price_layer1/BiasAdd?
price_layer1/ReluReluprice_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
price_layer1/Relu?
"price_layer2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"price_layer2/conv1d/ExpandDims/dim?
price_layer2/conv1d/ExpandDims
ExpandDimsprice_layer1/Relu:activations:0+price_layer2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2 
price_layer2/conv1d/ExpandDims?
/price_layer2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8price_layer2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/price_layer2/conv1d/ExpandDims_1/ReadVariableOp?
$price_layer2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$price_layer2/conv1d/ExpandDims_1/dim?
 price_layer2/conv1d/ExpandDims_1
ExpandDims7price_layer2/conv1d/ExpandDims_1/ReadVariableOp:value:0-price_layer2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2"
 price_layer2/conv1d/ExpandDims_1?
price_layer2/conv1dConv2D'price_layer2/conv1d/ExpandDims:output:0)price_layer2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
price_layer2/conv1d?
price_layer2/conv1d/SqueezeSqueezeprice_layer2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
price_layer2/conv1d/Squeeze?
#price_layer2/BiasAdd/ReadVariableOpReadVariableOp,price_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#price_layer2/BiasAdd/ReadVariableOp?
price_layer2/BiasAddBiasAdd$price_layer2/conv1d/Squeeze:output:0+price_layer2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
price_layer2/BiasAdd?
price_layer2/ReluReluprice_layer2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
price_layer2/Relu?
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dim?
average_pooling1d_2/ExpandDims
ExpandDimsprice_layer2/Relu:activations:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
average_pooling1d_2/ExpandDims?
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling1d_2/AvgPool?
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2
average_pooling1d_2/Squeeze{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
price_flatten/Const?
price_flatten/ReshapeReshape$average_pooling1d_2/Squeeze:output:0price_flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
price_flatten/Reshapev
concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_layer/concat/axis?
concat_layer/concatConcatV2price_flatten/Reshape:output:0inputs_1!concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????
2
concat_layer/concat?
"fixed_layer1/MatMul/ReadVariableOpReadVariableOp+fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"fixed_layer1/MatMul/ReadVariableOp?
fixed_layer1/MatMulMatMulconcat_layer/concat:output:0*fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer1/MatMul?
#fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp,fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#fixed_layer1/BiasAdd/ReadVariableOp?
fixed_layer1/BiasAddBiasAddfixed_layer1/MatMul:product:0+fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer1/BiasAdd
fixed_layer1/ReluRelufixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
fixed_layer1/Relu?
"fixed_layer2/MatMul/ReadVariableOpReadVariableOp+fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"fixed_layer2/MatMul/ReadVariableOp?
fixed_layer2/MatMulMatMulfixed_layer1/Relu:activations:0*fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer2/MatMul?
#fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp,fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#fixed_layer2/BiasAdd/ReadVariableOp?
fixed_layer2/BiasAddBiasAddfixed_layer2/MatMul:product:0+fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer2/BiasAdd
fixed_layer2/ReluRelufixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
fixed_layer2/Relu?
#action_output/MatMul/ReadVariableOpReadVariableOp,action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#action_output/MatMul/ReadVariableOp?
action_output/MatMulMatMulfixed_layer2/Relu:activations:0+action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
action_output/MatMul?
$action_output/BiasAdd/ReadVariableOpReadVariableOp-action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$action_output/BiasAdd/ReadVariableOp?
action_output/BiasAddBiasAddaction_output/MatMul:product:0,action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp$^price_layer1/BiasAdd/ReadVariableOp0^price_layer1/conv1d/ExpandDims_1/ReadVariableOp$^price_layer2/BiasAdd/ReadVariableOp0^price_layer2/conv1d/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2J
#price_layer1/BiasAdd/ReadVariableOp#price_layer1/BiasAdd/ReadVariableOp2b
/price_layer1/conv1d/ExpandDims_1/ReadVariableOp/price_layer1/conv1d/ExpandDims_1/ReadVariableOp2J
#price_layer2/BiasAdd/ReadVariableOp#price_layer2/BiasAdd/ReadVariableOp2b
/price_layer2/conv1d/ExpandDims_1/ReadVariableOp/price_layer2/conv1d/ExpandDims_1/ReadVariableOp:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
0__inference_fixed_layer2_layer_call_fn_356054143

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_3560536342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_price_flatten_layer_call_fn_356054090

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_3560535722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
"__inference__traced_save_356054234
file_prefix2
.savev2_price_layer1_kernel_read_readvariableop0
,savev2_price_layer1_bias_read_readvariableop2
.savev2_price_layer2_kernel_read_readvariableop0
,savev2_price_layer2_bias_read_readvariableop2
.savev2_fixed_layer1_kernel_read_readvariableop0
,savev2_fixed_layer1_bias_read_readvariableop2
.savev2_fixed_layer2_kernel_read_readvariableop0
,savev2_fixed_layer2_bias_read_readvariableop3
/savev2_action_output_kernel_read_readvariableop1
-savev2_action_output_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_price_layer1_kernel_read_readvariableop,savev2_price_layer1_bias_read_readvariableop.savev2_price_layer2_kernel_read_readvariableop,savev2_price_layer2_bias_read_readvariableop.savev2_fixed_layer1_kernel_read_readvariableop,savev2_fixed_layer1_bias_read_readvariableop.savev2_fixed_layer2_kernel_read_readvariableop,savev2_fixed_layer2_bias_read_readvariableop/savev2_action_output_kernel_read_readvariableop-savev2_action_output_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*{
_input_shapesj
h: : : : ::
:::::: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
+__inference_model_2_layer_call_fn_356054003
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_3560537472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
0__inference_fixed_layer1_layer_call_fn_356054123

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_3560536072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
u
K__inference_concat_layer_layer_call_and_return_conditional_losses_356053587

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????
2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
w
K__inference_concat_layer_layer_call_and_return_conditional_losses_356054097
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????
2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_356054134

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
S
7__inference_average_pooling1d_2_layer_call_fn_356053496

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_3560534902
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?(
?
F__inference_model_2_layer_call_and_return_conditional_losses_356053710
price_input
	env_input
price_layer1_356053681
price_layer1_356053683
price_layer2_356053686
price_layer2_356053688
fixed_layer1_356053694
fixed_layer1_356053696
fixed_layer2_356053699
fixed_layer2_356053701
action_output_356053704
action_output_356053706
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_356053681price_layer1_356053683*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_3560535172&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_356053686price_layer2_356053688*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_3560535492&
$price_layer2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_3560534902%
#average_pooling1d_2/PartitionedCall?
price_flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_3560535722
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0	env_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_3560535872
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_356053694fixed_layer1_356053696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_3560536072&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_356053699fixed_layer2_356053701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_3560536342&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_356053704action_output_356053706*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_action_output_layer_call_and_return_conditional_losses_3560536602'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?	
?
+__inference_model_2_layer_call_fn_356053829
price_input
	env_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_3560538062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_356053660

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?(
?
F__inference_model_2_layer_call_and_return_conditional_losses_356053677
price_input
	env_input
price_layer1_356053528
price_layer1_356053530
price_layer2_356053560
price_layer2_356053562
fixed_layer1_356053618
fixed_layer1_356053620
fixed_layer2_356053645
fixed_layer2_356053647
action_output_356053671
action_output_356053673
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_356053528price_layer1_356053530*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_3560535172&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_356053560price_layer2_356053562*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_3560535492&
$price_layer2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_3560534902%
#average_pooling1d_2/PartitionedCall?
price_flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_3560535722
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0	env_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_3560535872
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_356053618fixed_layer1_356053620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_3560536072&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_356053645fixed_layer2_356053647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_3560536342&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_356053671action_output_356053673*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_action_output_layer_call_and_return_conditional_losses_3560536602'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_356053549

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_356053517

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
'__inference_signature_wrapper_356053863
	env_input
price_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_3560534812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	env_input:XT
+
_output_shapes
:?????????
%
_user_specified_nameprice_input
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_356054153

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
+__inference_model_2_layer_call_fn_356053770
price_input
	env_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_3560537472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?	
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_356053607

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?(
?
F__inference_model_2_layer_call_and_return_conditional_losses_356053747

inputs
inputs_1
price_layer1_356053718
price_layer1_356053720
price_layer2_356053723
price_layer2_356053725
fixed_layer1_356053731
fixed_layer1_356053733
fixed_layer2_356053736
fixed_layer2_356053738
action_output_356053741
action_output_356053743
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_356053718price_layer1_356053720*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_3560535172&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_356053723price_layer2_356053725*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_3560535492&
$price_layer2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_3560534902%
#average_pooling1d_2/PartitionedCall?
price_flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_3560535722
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_3560535872
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_356053731fixed_layer1_356053733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_3560536072&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_356053736fixed_layer2_356053738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_3560536342&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_356053741action_output_356053743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_action_output_layer_call_and_return_conditional_losses_3560536602'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_356054114

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
0__inference_price_layer2_layer_call_fn_356054079

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_3560535492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
1__inference_action_output_layer_call_fn_356054162

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_action_output_layer_call_and_return_conditional_losses_3560536602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
+__inference_model_2_layer_call_fn_356054029
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_2_layer_call_and_return_conditional_losses_3560538062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_356053572

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?(
?
F__inference_model_2_layer_call_and_return_conditional_losses_356053806

inputs
inputs_1
price_layer1_356053777
price_layer1_356053779
price_layer2_356053782
price_layer2_356053784
fixed_layer1_356053790
fixed_layer1_356053792
fixed_layer2_356053795
fixed_layer2_356053797
action_output_356053800
action_output_356053802
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_356053777price_layer1_356053779*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_3560535172&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_356053782price_layer2_356053784*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_3560535492&
$price_layer2/StatefulPartitionedCall?
#average_pooling1d_2/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_3560534902%
#average_pooling1d_2/PartitionedCall?
price_flatten/PartitionedCallPartitionedCall,average_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_3560535722
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_3560535872
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_356053790fixed_layer1_356053792*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_3560536072&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_356053795fixed_layer2_356053797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_3560536342&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_356053800action_output_356053802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_action_output_layer_call_and_return_conditional_losses_3560536602'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?D
?
%__inference__traced_restore_356054292
file_prefix(
$assignvariableop_price_layer1_kernel(
$assignvariableop_1_price_layer1_bias*
&assignvariableop_2_price_layer2_kernel(
$assignvariableop_3_price_layer2_bias*
&assignvariableop_4_fixed_layer1_kernel(
$assignvariableop_5_fixed_layer1_bias*
&assignvariableop_6_fixed_layer2_kernel(
$assignvariableop_7_fixed_layer2_bias+
'assignvariableop_8_action_output_kernel)
%assignvariableop_9_action_output_bias 
assignvariableop_10_sgd_iter!
assignvariableop_11_sgd_decay)
%assignvariableop_12_sgd_learning_rate$
 assignvariableop_13_sgd_momentum
assignvariableop_14_total
assignvariableop_15_count
identity_17??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_price_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_price_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_price_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_price_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_fixed_layer1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_fixed_layer1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_fixed_layer2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_fixed_layer2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp'assignvariableop_8_action_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp%assignvariableop_9_action_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_sgd_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_sgd_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp%assignvariableop_12_sgd_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_sgd_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16?
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_356054070

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?P
?
F__inference_model_2_layer_call_and_return_conditional_losses_356053920
inputs_0
inputs_1<
8price_layer1_conv1d_expanddims_1_readvariableop_resource0
,price_layer1_biasadd_readvariableop_resource<
8price_layer2_conv1d_expanddims_1_readvariableop_resource0
,price_layer2_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?#price_layer1/BiasAdd/ReadVariableOp?/price_layer1/conv1d/ExpandDims_1/ReadVariableOp?#price_layer2/BiasAdd/ReadVariableOp?/price_layer2/conv1d/ExpandDims_1/ReadVariableOp?
"price_layer1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"price_layer1/conv1d/ExpandDims/dim?
price_layer1/conv1d/ExpandDims
ExpandDimsinputs_0+price_layer1/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
price_layer1/conv1d/ExpandDims?
/price_layer1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8price_layer1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/price_layer1/conv1d/ExpandDims_1/ReadVariableOp?
$price_layer1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$price_layer1/conv1d/ExpandDims_1/dim?
 price_layer1/conv1d/ExpandDims_1
ExpandDims7price_layer1/conv1d/ExpandDims_1/ReadVariableOp:value:0-price_layer1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2"
 price_layer1/conv1d/ExpandDims_1?
price_layer1/conv1dConv2D'price_layer1/conv1d/ExpandDims:output:0)price_layer1/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
price_layer1/conv1d?
price_layer1/conv1d/SqueezeSqueezeprice_layer1/conv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
price_layer1/conv1d/Squeeze?
#price_layer1/BiasAdd/ReadVariableOpReadVariableOp,price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#price_layer1/BiasAdd/ReadVariableOp?
price_layer1/BiasAddBiasAdd$price_layer1/conv1d/Squeeze:output:0+price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2
price_layer1/BiasAdd?
price_layer1/ReluReluprice_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
price_layer1/Relu?
"price_layer2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"price_layer2/conv1d/ExpandDims/dim?
price_layer2/conv1d/ExpandDims
ExpandDimsprice_layer1/Relu:activations:0+price_layer2/conv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? 2 
price_layer2/conv1d/ExpandDims?
/price_layer2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp8price_layer2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype021
/price_layer2/conv1d/ExpandDims_1/ReadVariableOp?
$price_layer2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2&
$price_layer2/conv1d/ExpandDims_1/dim?
 price_layer2/conv1d/ExpandDims_1
ExpandDims7price_layer2/conv1d/ExpandDims_1/ReadVariableOp:value:0-price_layer2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2"
 price_layer2/conv1d/ExpandDims_1?
price_layer2/conv1dConv2D'price_layer2/conv1d/ExpandDims:output:0)price_layer2/conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
price_layer2/conv1d?
price_layer2/conv1d/SqueezeSqueezeprice_layer2/conv1d:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims

?????????2
price_layer2/conv1d/Squeeze?
#price_layer2/BiasAdd/ReadVariableOpReadVariableOp,price_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#price_layer2/BiasAdd/ReadVariableOp?
price_layer2/BiasAddBiasAdd$price_layer2/conv1d/Squeeze:output:0+price_layer2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
price_layer2/BiasAdd?
price_layer2/ReluReluprice_layer2/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2
price_layer2/Relu?
"average_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"average_pooling1d_2/ExpandDims/dim?
average_pooling1d_2/ExpandDims
ExpandDimsprice_layer2/Relu:activations:0+average_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2 
average_pooling1d_2/ExpandDims?
average_pooling1d_2/AvgPoolAvgPool'average_pooling1d_2/ExpandDims:output:0*
T0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
average_pooling1d_2/AvgPool?
average_pooling1d_2/SqueezeSqueeze$average_pooling1d_2/AvgPool:output:0*
T0*+
_output_shapes
:?????????*
squeeze_dims
2
average_pooling1d_2/Squeeze{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
price_flatten/Const?
price_flatten/ReshapeReshape$average_pooling1d_2/Squeeze:output:0price_flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
price_flatten/Reshapev
concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_layer/concat/axis?
concat_layer/concatConcatV2price_flatten/Reshape:output:0inputs_1!concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????
2
concat_layer/concat?
"fixed_layer1/MatMul/ReadVariableOpReadVariableOp+fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02$
"fixed_layer1/MatMul/ReadVariableOp?
fixed_layer1/MatMulMatMulconcat_layer/concat:output:0*fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer1/MatMul?
#fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp,fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#fixed_layer1/BiasAdd/ReadVariableOp?
fixed_layer1/BiasAddBiasAddfixed_layer1/MatMul:product:0+fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer1/BiasAdd
fixed_layer1/ReluRelufixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
fixed_layer1/Relu?
"fixed_layer2/MatMul/ReadVariableOpReadVariableOp+fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"fixed_layer2/MatMul/ReadVariableOp?
fixed_layer2/MatMulMatMulfixed_layer1/Relu:activations:0*fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer2/MatMul?
#fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp,fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#fixed_layer2/BiasAdd/ReadVariableOp?
fixed_layer2/BiasAddBiasAddfixed_layer2/MatMul:product:0+fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
fixed_layer2/BiasAdd
fixed_layer2/ReluRelufixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
fixed_layer2/Relu?
#action_output/MatMul/ReadVariableOpReadVariableOp,action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02%
#action_output/MatMul/ReadVariableOp?
action_output/MatMulMatMulfixed_layer2/Relu:activations:0+action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
action_output/MatMul?
$action_output/BiasAdd/ReadVariableOpReadVariableOp-action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$action_output/BiasAdd/ReadVariableOp?
action_output/BiasAddBiasAddaction_output/MatMul:product:0,action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp$^price_layer1/BiasAdd/ReadVariableOp0^price_layer1/conv1d/ExpandDims_1/ReadVariableOp$^price_layer2/BiasAdd/ReadVariableOp0^price_layer2/conv1d/ExpandDims_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:?????????:?????????::::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2J
#price_layer1/BiasAdd/ReadVariableOp#price_layer1/BiasAdd/ReadVariableOp2b
/price_layer1/conv1d/ExpandDims_1/ReadVariableOp/price_layer1/conv1d/ExpandDims_1/ReadVariableOp2J
#price_layer2/BiasAdd/ReadVariableOp#price_layer2/BiasAdd/ReadVariableOp2b
/price_layer2/conv1d/ExpandDims_1/ReadVariableOp/price_layer2/conv1d/ExpandDims_1/ReadVariableOp:U Q
+
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
\
0__inference_concat_layer_layer_call_fn_356054103
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_3560535872
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_356054045

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv1d/ExpandDims/dim?
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????2
conv1d/ExpandDims?
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim?
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1?
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv1d?
conv1d/SqueezeSqueezeconv1d:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

?????????2
conv1d/Squeeze?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_356053490

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2

ExpandDims?
AvgPoolAvgPoolExpandDims:output:0*
T0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool?
SqueezeSqueezeAvgPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
2	
Squeezez
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
0__inference_price_layer1_layer_call_fn_356054054

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_3560535172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_356053634

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	env_input2
serving_default_env_input:0?????????
G
price_input8
serving_default_price_input:0?????????A
action_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?P
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
	optimizer
loss
	variables
regularization_losses
trainable_variables
	keras_api

signatures
r__call__
*s&call_and_return_all_conditional_losses
t_default_save_signature"?M
_tf_keras_network?M{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_2", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["average_pooling1d_2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5, 1]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "Conv1D", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Conv1D", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "AveragePooling1D", "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "name": "average_pooling1d_2", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["average_pooling1d_2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}}, "training_config": {"loss": {"action_output": "mse"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "price_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}}
?	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "price_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv1D", "name": "price_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [2]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 32]}}
?
	variables
regularization_losses
 trainable_variables
!	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "AveragePooling1D", "name": "average_pooling1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling1d_2", "trainable": true, "dtype": "float32", "strides": {"class_name": "__tuple__", "items": [2]}, "pool_size": {"class_name": "__tuple__", "items": [2]}, "padding": "valid", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
"	variables
#regularization_losses
$trainable_variables
%	keras_api
{__call__
*|&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "price_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "env_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}}
?
&	variables
'regularization_losses
(trainable_variables
)	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 8]}, {"class_name": "TensorShape", "items": [null, 2]}]}
?

*kernel
+bias
,	variables
-regularization_losses
.trainable_variables
/	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

6kernel
7bias
8	variables
9regularization_losses
:trainable_variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "action_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
I
<iter
	=decay
>learning_rate
?momentum"
	optimizer
 "
trackable_dict_wrapper
f
0
1
2
3
*4
+5
06
17
68
79"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
*4
+5
06
17
68
79"
trackable_list_wrapper
?
@metrics
	variables
Anon_trainable_variables
Blayer_metrics

Clayers
Dlayer_regularization_losses
regularization_losses
trainable_variables
r__call__
t_default_save_signature
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):' 2price_layer1/kernel
: 2price_layer1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Emetrics
	variables
Fnon_trainable_variables

Glayers
regularization_losses
Hlayer_regularization_losses
Ilayer_metrics
trainable_variables
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
):' 2price_layer2/kernel
:2price_layer2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Jmetrics
	variables
Knon_trainable_variables

Llayers
regularization_losses
Mlayer_regularization_losses
Nlayer_metrics
trainable_variables
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ometrics
	variables
Pnon_trainable_variables

Qlayers
regularization_losses
Rlayer_regularization_losses
Slayer_metrics
 trainable_variables
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tmetrics
"	variables
Unon_trainable_variables

Vlayers
#regularization_losses
Wlayer_regularization_losses
Xlayer_metrics
$trainable_variables
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ymetrics
&	variables
Znon_trainable_variables

[layers
'regularization_losses
\layer_regularization_losses
]layer_metrics
(trainable_variables
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
%:#
2fixed_layer1/kernel
:2fixed_layer1/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
^metrics
,	variables
_non_trainable_variables

`layers
-regularization_losses
alayer_regularization_losses
blayer_metrics
.trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2fixed_layer2/kernel
:2fixed_layer2/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
cmetrics
2	variables
dnon_trainable_variables

elayers
3regularization_losses
flayer_regularization_losses
glayer_metrics
4trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2action_output/kernel
 :2action_output/bias
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
hmetrics
8	variables
inon_trainable_variables

jlayers
9regularization_losses
klayer_regularization_losses
llayer_metrics
:trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
'
m0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
	ntotal
	ocount
p	variables
q	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
n0
o1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
?2?
+__inference_model_2_layer_call_fn_356054003
+__inference_model_2_layer_call_fn_356054029
+__inference_model_2_layer_call_fn_356053770
+__inference_model_2_layer_call_fn_356053829?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
F__inference_model_2_layer_call_and_return_conditional_losses_356053920
F__inference_model_2_layer_call_and_return_conditional_losses_356053710
F__inference_model_2_layer_call_and_return_conditional_losses_356053677
F__inference_model_2_layer_call_and_return_conditional_losses_356053977?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference__wrapped_model_356053481?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *X?U
S?P
)?&
price_input?????????
#? 
	env_input?????????
?2?
0__inference_price_layer1_layer_call_fn_356054054?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_price_layer1_layer_call_and_return_conditional_losses_356054045?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_price_layer2_layer_call_fn_356054079?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_price_layer2_layer_call_and_return_conditional_losses_356054070?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
7__inference_average_pooling1d_2_layer_call_fn_356053496?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_356053490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *3?0
.?+'???????????????????????????
?2?
1__inference_price_flatten_layer_call_fn_356054090?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_price_flatten_layer_call_and_return_conditional_losses_356054085?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_concat_layer_layer_call_fn_356054103?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_concat_layer_layer_call_and_return_conditional_losses_356054097?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_fixed_layer1_layer_call_fn_356054123?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_356054114?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_fixed_layer2_layer_call_fn_356054143?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_356054134?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_action_output_layer_call_fn_356054162?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_action_output_layer_call_and_return_conditional_losses_356054153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
'__inference_signature_wrapper_356053863	env_inputprice_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
$__inference__wrapped_model_356053481?
*+0167b?_
X?U
S?P
)?&
price_input?????????
#? 
	env_input?????????
? "=?:
8
action_output'?$
action_output??????????
L__inference_action_output_layer_call_and_return_conditional_losses_356054153\67/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_action_output_layer_call_fn_356054162O67/?,
%?"
 ?
inputs?????????
? "???????????
R__inference_average_pooling1d_2_layer_call_and_return_conditional_losses_356053490?E?B
;?8
6?3
inputs'???????????????????????????
? ";?8
1?.
0'???????????????????????????
? ?
7__inference_average_pooling1d_2_layer_call_fn_356053496wE?B
;?8
6?3
inputs'???????????????????????????
? ".?+'????????????????????????????
K__inference_concat_layer_layer_call_and_return_conditional_losses_356054097?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????

? ?
0__inference_concat_layer_layer_call_fn_356054103vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_356054114\*+/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ?
0__inference_fixed_layer1_layer_call_fn_356054123O*+/?,
%?"
 ?
inputs?????????

? "???????????
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_356054134\01/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
0__inference_fixed_layer2_layer_call_fn_356054143O01/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_model_2_layer_call_and_return_conditional_losses_356053677?
*+0167j?g
`?]
S?P
)?&
price_input?????????
#? 
	env_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_model_2_layer_call_and_return_conditional_losses_356053710?
*+0167j?g
`?]
S?P
)?&
price_input?????????
#? 
	env_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_model_2_layer_call_and_return_conditional_losses_356053920?
*+0167f?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_model_2_layer_call_and_return_conditional_losses_356053977?
*+0167f?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
+__inference_model_2_layer_call_fn_356053770?
*+0167j?g
`?]
S?P
)?&
price_input?????????
#? 
	env_input?????????
p

 
? "???????????
+__inference_model_2_layer_call_fn_356053829?
*+0167j?g
`?]
S?P
)?&
price_input?????????
#? 
	env_input?????????
p 

 
? "???????????
+__inference_model_2_layer_call_fn_356054003?
*+0167f?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
+__inference_model_2_layer_call_fn_356054029?
*+0167f?c
\?Y
O?L
&?#
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
L__inference_price_flatten_layer_call_and_return_conditional_losses_356054085\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_price_flatten_layer_call_fn_356054090O3?0
)?&
$?!
inputs?????????
? "???????????
K__inference_price_layer1_layer_call_and_return_conditional_losses_356054045d3?0
)?&
$?!
inputs?????????
? ")?&
?
0????????? 
? ?
0__inference_price_layer1_layer_call_fn_356054054W3?0
)?&
$?!
inputs?????????
? "?????????? ?
K__inference_price_layer2_layer_call_and_return_conditional_losses_356054070d3?0
)?&
$?!
inputs????????? 
? ")?&
?
0?????????
? ?
0__inference_price_layer2_layer_call_fn_356054079W3?0
)?&
$?!
inputs????????? 
? "???????????
'__inference_signature_wrapper_356053863?
*+0167y?v
? 
o?l
0
	env_input#? 
	env_input?????????
8
price_input)?&
price_input?????????"=?:
8
action_output'?$
action_output?????????