??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
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
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8ɮ
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
?
price_layer1/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nameprice_layer1/lstm_cell/kernel
?
1price_layer1/lstm_cell/kernel/Read/ReadVariableOpReadVariableOpprice_layer1/lstm_cell/kernel*
_output_shapes

: *
dtype0
?
'price_layer1/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *8
shared_name)'price_layer1/lstm_cell/recurrent_kernel
?
;price_layer1/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp'price_layer1/lstm_cell/recurrent_kernel*
_output_shapes

: *
dtype0
?
price_layer1/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nameprice_layer1/lstm_cell/bias
?
/price_layer1/lstm_cell/bias/Read/ReadVariableOpReadVariableOpprice_layer1/lstm_cell/bias*
_output_shapes
: *
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
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?"B?" B?"
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		optimizer

loss
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
6
0iter
	1decay
2learning_rate
3momentum
 
 
?
40
51
62
3
4
$5
%6
*7
+8
?
40
51
62
3
4
$5
%6
*7
+8
?
7metrics
8layer_regularization_losses

9layers
:non_trainable_variables
regularization_losses
	variables
;layer_metrics
trainable_variables
 
~

4kernel
5recurrent_kernel
6bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
 
 

40
51
62

40
51
62
?
@metrics
Alayer_regularization_losses

Blayers
Cnon_trainable_variables
regularization_losses
trainable_variables
	variables
Dlayer_metrics

Estates
 
 
 
?
Flayer_regularization_losses
Gmetrics

Hlayers
Inon_trainable_variables
regularization_losses
	variables
Jlayer_metrics
trainable_variables
 
 
 
?
Klayer_regularization_losses
Lmetrics

Mlayers
Nnon_trainable_variables
regularization_losses
	variables
Olayer_metrics
trainable_variables
_]
VARIABLE_VALUEfixed_layer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Player_regularization_losses
Qmetrics

Rlayers
Snon_trainable_variables
 regularization_losses
!	variables
Tlayer_metrics
"trainable_variables
_]
VARIABLE_VALUEfixed_layer2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
?
Ulayer_regularization_losses
Vmetrics

Wlayers
Xnon_trainable_variables
&regularization_losses
'	variables
Ylayer_metrics
(trainable_variables
`^
VARIABLE_VALUEaction_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEaction_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
?
Zlayer_regularization_losses
[metrics

\layers
]non_trainable_variables
,regularization_losses
-	variables
^layer_metrics
.trainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEprice_layer1/lstm_cell/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'price_layer1/lstm_cell/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEprice_layer1/lstm_cell/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE

_0
 
8
0
1
2
3
4
5
6
7
 
 
 

40
51
62

40
51
62
?
`layer_regularization_losses
ametrics

blayers
cnon_trainable_variables
<regularization_losses
=	variables
dlayer_metrics
>trainable_variables
 
 

0
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
	etotal
	fcount
g	variables
h	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

g	variables
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_env_inputserving_default_price_inputprice_layer1/lstm_cell/kernel'price_layer1/lstm_cell/recurrent_kernelprice_layer1/lstm_cell/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_117626529
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'fixed_layer1/kernel/Read/ReadVariableOp%fixed_layer1/bias/Read/ReadVariableOp'fixed_layer2/kernel/Read/ReadVariableOp%fixed_layer2/bias/Read/ReadVariableOp(action_output/kernel/Read/ReadVariableOp&action_output/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp1price_layer1/lstm_cell/kernel/Read/ReadVariableOp;price_layer1/lstm_cell/recurrent_kernel/Read/ReadVariableOp/price_layer1/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
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
"__inference__traced_save_117627841
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumprice_layer1/lstm_cell/kernel'price_layer1/lstm_cell/recurrent_kernelprice_layer1/lstm_cell/biastotalcount*
Tin
2*
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
%__inference__traced_restore_117627896??
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_117626344

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
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117625373

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?D
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117625901

inputs
lstm_cell_117625819
lstm_cell_117625821
lstm_cell_117625823
identity??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_117625819lstm_cell_117625821lstm_cell_117625823*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_layer_call_and_return_conditional_losses_1176254062#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_117625819lstm_cell_117625821lstm_cell_117625823*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117625832* 
condR
while_cond_117625831*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?	
?
)__inference_model_layer_call_fn_117626909
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
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1176264232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::22
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
1__inference_action_output_layer_call_fn_117627672

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
L__inference_action_output_layer_call_and_return_conditional_losses_1176263442
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
?$
?
while_body_117625832
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_117625856_0
while_lstm_cell_117625858_0
while_lstm_cell_117625860_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_117625856
while_lstm_cell_117625858
while_lstm_cell_117625860??'while/lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_117625856_0while_lstm_cell_117625858_0while_lstm_cell_117625860_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_layer_call_and_return_conditional_losses_1176254062)
'while/lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_117625856while_lstm_cell_117625856_0"8
while_lstm_cell_117625858while_lstm_cell_117625858_0"8
while_lstm_cell_117625860while_lstm_cell_117625860_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
0__inference_price_layer1_layer_call_fn_117627250

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176260672
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_fixed_layer2_layer_call_fn_117627653

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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_1176263182
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
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_117627595

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
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?A
?
%__inference__traced_restore_117627896
file_prefix(
$assignvariableop_fixed_layer1_kernel(
$assignvariableop_1_fixed_layer1_bias*
&assignvariableop_2_fixed_layer2_kernel(
$assignvariableop_3_fixed_layer2_bias+
'assignvariableop_4_action_output_kernel)
%assignvariableop_5_action_output_bias
assignvariableop_6_sgd_iter 
assignvariableop_7_sgd_decay(
$assignvariableop_8_sgd_learning_rate#
assignvariableop_9_sgd_momentum5
1assignvariableop_10_price_layer1_lstm_cell_kernel?
;assignvariableop_11_price_layer1_lstm_cell_recurrent_kernel3
/assignvariableop_12_price_layer1_lstm_cell_bias
assignvariableop_13_total
assignvariableop_14_count
identity_16??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_fixed_layer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_fixed_layer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_fixed_layer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_fixed_layer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp'assignvariableop_4_action_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp%assignvariableop_5_action_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp1assignvariableop_10_price_layer1_lstm_cell_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp;assignvariableop_11_price_layer1_lstm_cell_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp/assignvariableop_12_price_layer1_lstm_cell_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_149
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_15?
Identity_16IdentityIdentity_15:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_16"#
identity_16Identity_16:output:0*Q
_input_shapes@
>: :::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
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
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_117626256

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
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?@
?
while_body_117626135
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource??&while/lstm_cell/BiasAdd/ReadVariableOp?%while/lstm_cell/MatMul/ReadVariableOp?'while/lstm_cell/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul?
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp?
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul_1?
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/add?
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp?
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell/split?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul?
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu?
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_1?
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu_1?
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?Y
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627239

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117627154* 
condR
while_cond_117627153*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
while_body_117625700
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_117625724_0
while_lstm_cell_117625726_0
while_lstm_cell_117625728_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_117625724
while_lstm_cell_117625726
while_lstm_cell_117625728??'while/lstm_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_117625724_0while_lstm_cell_117625726_0while_lstm_cell_117625728_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_layer_call_and_return_conditional_losses_1176253732)
'while/lstm_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_117625724while_lstm_cell_117625724_0"8
while_lstm_cell_117625726while_lstm_cell_117625726_0"8
while_lstm_cell_117625728while_lstm_cell_117625728_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?
?
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117625406

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul_1/ReadVariableOpy
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_namestates:OK
'
_output_shapes
:?????????
 
_user_specified_namestates
?
M
1__inference_price_flatten_layer_call_fn_117627600

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
L__inference_price_flatten_layer_call_and_return_conditional_losses_1176262562
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117627738

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117627705

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?MatMul_1/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul_1/ReadVariableOp{
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2

MatMul_1k
addAddV2MatMul:product:0MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim?
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:?????????2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:?????????2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:?????????2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:?????????2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:?????????2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:?????????2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:?????????2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
while_cond_117625981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117625981___redundant_placeholder07
3while_while_cond_117625981___redundant_placeholder17
3while_while_cond_117625981___redundant_placeholder27
3while_while_cond_117625981___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
0__inference_price_layer1_layer_call_fn_117627578
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176257692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
0__inference_price_layer1_layer_call_fn_117627261

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176262202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
\
0__inference_concat_layer_layer_call_fn_117627613
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_1176262712
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
?	
?
'__inference_signature_wrapper_117626529
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
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_1176253002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::22
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
?@
?
while_body_117625982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource??&while/lstm_cell/BiasAdd/ReadVariableOp?%while/lstm_cell/MatMul/ReadVariableOp?'while/lstm_cell/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul?
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp?
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul_1?
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/add?
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp?
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell/split?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul?
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu?
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_1?
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu_1?
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?T
?
!price_layer1_while_body_1176267766
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_35
1price_layer1_while_price_layer1_strided_slice_1_0q
mprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0A
=price_layer1_while_lstm_cell_matmul_readvariableop_resource_0C
?price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0B
>price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0
price_layer1_while_identity!
price_layer1_while_identity_1!
price_layer1_while_identity_2!
price_layer1_while_identity_3!
price_layer1_while_identity_4!
price_layer1_while_identity_53
/price_layer1_while_price_layer1_strided_slice_1o
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor?
;price_layer1_while_lstm_cell_matmul_readvariableop_resourceA
=price_layer1_while_lstm_cell_matmul_1_readvariableop_resource@
<price_layer1_while_lstm_cell_biasadd_readvariableop_resource??3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp?2price_layer1/while/lstm_cell/MatMul/ReadVariableOp?4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp?
Dprice_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dprice_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6price_layer1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0price_layer1_while_placeholderMprice_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6price_layer1/while/TensorArrayV2Read/TensorListGetItem?
2price_layer1/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp=price_layer1_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype024
2price_layer1/while/lstm_cell/MatMul/ReadVariableOp?
#price_layer1/while/lstm_cell/MatMulMatMul=price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0:price_layer1/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#price_layer1/while/lstm_cell/MatMul?
4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype026
4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp?
%price_layer1/while/lstm_cell/MatMul_1MatMul price_layer1_while_placeholder_2<price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell/MatMul_1?
 price_layer1/while/lstm_cell/addAddV2-price_layer1/while/lstm_cell/MatMul:product:0/price_layer1/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2"
 price_layer1/while/lstm_cell/add?
3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype025
3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp?
$price_layer1/while/lstm_cell/BiasAddBiasAdd$price_layer1/while/lstm_cell/add:z:0;price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$price_layer1/while/lstm_cell/BiasAdd?
"price_layer1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"price_layer1/while/lstm_cell/Const?
,price_layer1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,price_layer1/while/lstm_cell/split/split_dim?
"price_layer1/while/lstm_cell/splitSplit5price_layer1/while/lstm_cell/split/split_dim:output:0-price_layer1/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2$
"price_layer1/while/lstm_cell/split?
$price_layer1/while/lstm_cell/SigmoidSigmoid+price_layer1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell/Sigmoid?
&price_layer1/while/lstm_cell/Sigmoid_1Sigmoid+price_layer1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2(
&price_layer1/while/lstm_cell/Sigmoid_1?
 price_layer1/while/lstm_cell/mulMul*price_layer1/while/lstm_cell/Sigmoid_1:y:0 price_layer1_while_placeholder_3*
T0*'
_output_shapes
:?????????2"
 price_layer1/while/lstm_cell/mul?
!price_layer1/while/lstm_cell/ReluRelu+price_layer1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2#
!price_layer1/while/lstm_cell/Relu?
"price_layer1/while/lstm_cell/mul_1Mul(price_layer1/while/lstm_cell/Sigmoid:y:0/price_layer1/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell/mul_1?
"price_layer1/while/lstm_cell/add_1AddV2$price_layer1/while/lstm_cell/mul:z:0&price_layer1/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell/add_1?
&price_layer1/while/lstm_cell/Sigmoid_2Sigmoid+price_layer1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2(
&price_layer1/while/lstm_cell/Sigmoid_2?
#price_layer1/while/lstm_cell/Relu_1Relu&price_layer1/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2%
#price_layer1/while/lstm_cell/Relu_1?
"price_layer1/while/lstm_cell/mul_2Mul*price_layer1/while/lstm_cell/Sigmoid_2:y:01price_layer1/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell/mul_2?
7price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer1_while_placeholder_1price_layer1_while_placeholder&price_layer1/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype029
7price_layer1/while/TensorArrayV2Write/TensorListSetItemv
price_layer1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/while/add/y?
price_layer1/while/addAddV2price_layer1_while_placeholder!price_layer1/while/add/y:output:0*
T0*
_output_shapes
: 2
price_layer1/while/addz
price_layer1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/while/add_1/y?
price_layer1/while/add_1AddV22price_layer1_while_price_layer1_while_loop_counter#price_layer1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
price_layer1/while/add_1?
price_layer1/while/IdentityIdentityprice_layer1/while/add_1:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity?
price_layer1/while/Identity_1Identity8price_layer1_while_price_layer1_while_maximum_iterations4^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_1?
price_layer1/while/Identity_2Identityprice_layer1/while/add:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_2?
price_layer1/while/Identity_3IdentityGprice_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_3?
price_layer1/while/Identity_4Identity&price_layer1/while/lstm_cell/mul_2:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_4?
price_layer1/while/Identity_5Identity&price_layer1/while/lstm_cell/add_1:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_5"C
price_layer1_while_identity$price_layer1/while/Identity:output:0"G
price_layer1_while_identity_1&price_layer1/while/Identity_1:output:0"G
price_layer1_while_identity_2&price_layer1/while/Identity_2:output:0"G
price_layer1_while_identity_3&price_layer1/while/Identity_3:output:0"G
price_layer1_while_identity_4&price_layer1/while/Identity_4:output:0"G
price_layer1_while_identity_5&price_layer1/while/Identity_5:output:0"~
<price_layer1_while_lstm_cell_biasadd_readvariableop_resource>price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0"?
=price_layer1_while_lstm_cell_matmul_1_readvariableop_resource?price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0"|
;price_layer1_while_lstm_cell_matmul_readvariableop_resource=price_layer1_while_lstm_cell_matmul_readvariableop_resource_0"d
/price_layer1_while_price_layer1_strided_slice_11price_layer1_while_price_layer1_strided_slice_1_0"?
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensormprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2j
3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp2h
2price_layer1/while/lstm_cell/MatMul/ReadVariableOp2price_layer1/while/lstm_cell/MatMul/ReadVariableOp2l
4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?Y
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117626067

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117625982* 
condR
while_cond_117625981*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_117627644

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
?@
?
while_body_117627154
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource??&while/lstm_cell/BiasAdd/ReadVariableOp?%while/lstm_cell/MatMul/ReadVariableOp?'while/lstm_cell/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul?
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp?
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul_1?
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/add?
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp?
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell/split?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul?
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu?
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_1?
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu_1?
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?D
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117625769

inputs
lstm_cell_117625687
lstm_cell_117625689
lstm_cell_117625691
identity??!lstm_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_117625687lstm_cell_117625689lstm_cell_117625691*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_layer_call_and_return_conditional_losses_1176253732#
!lstm_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_117625687lstm_cell_117625689lstm_cell_117625691*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117625700* 
condR
while_cond_117625699*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0"^lstm_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_117625831
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117625831___redundant_placeholder07
3while_while_cond_117625831___redundant_placeholder17
3while_while_cond_117625831___redundant_placeholder27
3while_while_cond_117625831___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?@
?
while_body_117627001
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource??&while/lstm_cell/BiasAdd/ReadVariableOp?%while/lstm_cell/MatMul/ReadVariableOp?'while/lstm_cell/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul?
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp?
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul_1?
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/add?
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp?
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell/split?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul?
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu?
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_1?
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu_1?
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
??
?
D__inference_model_layer_call_and_return_conditional_losses_117626707
inputs_0
inputs_19
5price_layer1_lstm_cell_matmul_readvariableop_resource;
7price_layer1_lstm_cell_matmul_1_readvariableop_resource:
6price_layer1_lstm_cell_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?-price_layer1/lstm_cell/BiasAdd/ReadVariableOp?,price_layer1/lstm_cell/MatMul/ReadVariableOp?.price_layer1/lstm_cell/MatMul_1/ReadVariableOp?price_layer1/while`
price_layer1/ShapeShapeinputs_0*
T0*
_output_shapes
:2
price_layer1/Shape?
 price_layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 price_layer1/strided_slice/stack?
"price_layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer1/strided_slice/stack_1?
"price_layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer1/strided_slice/stack_2?
price_layer1/strided_sliceStridedSliceprice_layer1/Shape:output:0)price_layer1/strided_slice/stack:output:0+price_layer1/strided_slice/stack_1:output:0+price_layer1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer1/strided_slicev
price_layer1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros/mul/y?
price_layer1/zeros/mulMul#price_layer1/strided_slice:output:0!price_layer1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros/muly
price_layer1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer1/zeros/Less/y?
price_layer1/zeros/LessLessprice_layer1/zeros/mul:z:0"price_layer1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros/Less|
price_layer1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros/packed/1?
price_layer1/zeros/packedPack#price_layer1/strided_slice:output:0$price_layer1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer1/zeros/packedy
price_layer1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer1/zeros/Const?
price_layer1/zerosFill"price_layer1/zeros/packed:output:0!price_layer1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/zerosz
price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros_1/mul/y?
price_layer1/zeros_1/mulMul#price_layer1/strided_slice:output:0#price_layer1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros_1/mul}
price_layer1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer1/zeros_1/Less/y?
price_layer1/zeros_1/LessLessprice_layer1/zeros_1/mul:z:0$price_layer1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros_1/Less?
price_layer1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros_1/packed/1?
price_layer1/zeros_1/packedPack#price_layer1/strided_slice:output:0&price_layer1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer1/zeros_1/packed}
price_layer1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer1/zeros_1/Const?
price_layer1/zeros_1Fill$price_layer1/zeros_1/packed:output:0#price_layer1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/zeros_1?
price_layer1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer1/transpose/perm?
price_layer1/transpose	Transposeinputs_0$price_layer1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
price_layer1/transposev
price_layer1/Shape_1Shapeprice_layer1/transpose:y:0*
T0*
_output_shapes
:2
price_layer1/Shape_1?
"price_layer1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer1/strided_slice_1/stack?
$price_layer1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_1/stack_1?
$price_layer1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_1/stack_2?
price_layer1/strided_slice_1StridedSliceprice_layer1/Shape_1:output:0+price_layer1/strided_slice_1/stack:output:0-price_layer1/strided_slice_1/stack_1:output:0-price_layer1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer1/strided_slice_1?
(price_layer1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(price_layer1/TensorArrayV2/element_shape?
price_layer1/TensorArrayV2TensorListReserve1price_layer1/TensorArrayV2/element_shape:output:0%price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer1/TensorArrayV2?
Bprice_layer1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bprice_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape?
4price_layer1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorprice_layer1/transpose:y:0Kprice_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4price_layer1/TensorArrayUnstack/TensorListFromTensor?
"price_layer1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer1/strided_slice_2/stack?
$price_layer1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_2/stack_1?
$price_layer1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_2/stack_2?
price_layer1/strided_slice_2StridedSliceprice_layer1/transpose:y:0+price_layer1/strided_slice_2/stack:output:0-price_layer1/strided_slice_2/stack_1:output:0-price_layer1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
price_layer1/strided_slice_2?
,price_layer1/lstm_cell/MatMul/ReadVariableOpReadVariableOp5price_layer1_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,price_layer1/lstm_cell/MatMul/ReadVariableOp?
price_layer1/lstm_cell/MatMulMatMul%price_layer1/strided_slice_2:output:04price_layer1/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell/MatMul?
.price_layer1/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7price_layer1_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype020
.price_layer1/lstm_cell/MatMul_1/ReadVariableOp?
price_layer1/lstm_cell/MatMul_1MatMulprice_layer1/zeros:output:06price_layer1/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell/MatMul_1?
price_layer1/lstm_cell/addAddV2'price_layer1/lstm_cell/MatMul:product:0)price_layer1/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell/add?
-price_layer1/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6price_layer1_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-price_layer1/lstm_cell/BiasAdd/ReadVariableOp?
price_layer1/lstm_cell/BiasAddBiasAddprice_layer1/lstm_cell/add:z:05price_layer1/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
price_layer1/lstm_cell/BiasAdd~
price_layer1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/lstm_cell/Const?
&price_layer1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&price_layer1/lstm_cell/split/split_dim?
price_layer1/lstm_cell/splitSplit/price_layer1/lstm_cell/split/split_dim:output:0'price_layer1/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
price_layer1/lstm_cell/split?
price_layer1/lstm_cell/SigmoidSigmoid%price_layer1/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell/Sigmoid?
 price_layer1/lstm_cell/Sigmoid_1Sigmoid%price_layer1/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2"
 price_layer1/lstm_cell/Sigmoid_1?
price_layer1/lstm_cell/mulMul$price_layer1/lstm_cell/Sigmoid_1:y:0price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/mul?
price_layer1/lstm_cell/ReluRelu%price_layer1/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/Relu?
price_layer1/lstm_cell/mul_1Mul"price_layer1/lstm_cell/Sigmoid:y:0)price_layer1/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/mul_1?
price_layer1/lstm_cell/add_1AddV2price_layer1/lstm_cell/mul:z:0 price_layer1/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/add_1?
 price_layer1/lstm_cell/Sigmoid_2Sigmoid%price_layer1/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2"
 price_layer1/lstm_cell/Sigmoid_2?
price_layer1/lstm_cell/Relu_1Relu price_layer1/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/Relu_1?
price_layer1/lstm_cell/mul_2Mul$price_layer1/lstm_cell/Sigmoid_2:y:0+price_layer1/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/mul_2?
*price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*price_layer1/TensorArrayV2_1/element_shape?
price_layer1/TensorArrayV2_1TensorListReserve3price_layer1/TensorArrayV2_1/element_shape:output:0%price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer1/TensorArrayV2_1h
price_layer1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer1/time?
%price_layer1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%price_layer1/while/maximum_iterations?
price_layer1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
price_layer1/while/loop_counter?
price_layer1/whileWhile(price_layer1/while/loop_counter:output:0.price_layer1/while/maximum_iterations:output:0price_layer1/time:output:0%price_layer1/TensorArrayV2_1:handle:0price_layer1/zeros:output:0price_layer1/zeros_1:output:0%price_layer1/strided_slice_1:output:0Dprice_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:05price_layer1_lstm_cell_matmul_readvariableop_resource7price_layer1_lstm_cell_matmul_1_readvariableop_resource6price_layer1_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!price_layer1_while_body_117626598*-
cond%R#
!price_layer1_while_cond_117626597*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
price_layer1/while?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStackprice_layer1/while:output:3Fprice_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype021
/price_layer1/TensorArrayV2Stack/TensorListStack?
"price_layer1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"price_layer1/strided_slice_3/stack?
$price_layer1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$price_layer1/strided_slice_3/stack_1?
$price_layer1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_3/stack_2?
price_layer1/strided_slice_3StridedSlice8price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0+price_layer1/strided_slice_3/stack:output:0-price_layer1/strided_slice_3/stack_1:output:0-price_layer1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
price_layer1/strided_slice_3?
price_layer1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer1/transpose_1/perm?
price_layer1/transpose_1	Transpose8price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0&price_layer1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
price_layer1/transpose_1?
price_layer1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer1/runtime{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
price_flatten/Const?
price_flatten/ReshapeReshape%price_layer1/strided_slice_3:output:0price_flatten/Const:output:0*
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
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp.^price_layer1/lstm_cell/BiasAdd/ReadVariableOp-^price_layer1/lstm_cell/MatMul/ReadVariableOp/^price_layer1/lstm_cell/MatMul_1/ReadVariableOp^price_layer1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2^
-price_layer1/lstm_cell/BiasAdd/ReadVariableOp-price_layer1/lstm_cell/BiasAdd/ReadVariableOp2\
,price_layer1/lstm_cell/MatMul/ReadVariableOp,price_layer1/lstm_cell/MatMul/ReadVariableOp2`
.price_layer1/lstm_cell/MatMul_1/ReadVariableOp.price_layer1/lstm_cell/MatMul_1/ReadVariableOp2(
price_layer1/whileprice_layer1/while:U Q
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
?
?
-__inference_lstm_cell_layer_call_fn_117627755

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_layer_call_and_return_conditional_losses_1176253732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?
?
while_cond_117627153
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117627153___redundant_placeholder07
3while_while_cond_117627153___redundant_placeholder17
3while_while_cond_117627153___redundant_placeholder27
3while_while_cond_117627153___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
'model_price_layer1_while_cond_117625190B
>model_price_layer1_while_model_price_layer1_while_loop_counterH
Dmodel_price_layer1_while_model_price_layer1_while_maximum_iterations(
$model_price_layer1_while_placeholder*
&model_price_layer1_while_placeholder_1*
&model_price_layer1_while_placeholder_2*
&model_price_layer1_while_placeholder_3D
@model_price_layer1_while_less_model_price_layer1_strided_slice_1]
Ymodel_price_layer1_while_model_price_layer1_while_cond_117625190___redundant_placeholder0]
Ymodel_price_layer1_while_model_price_layer1_while_cond_117625190___redundant_placeholder1]
Ymodel_price_layer1_while_model_price_layer1_while_cond_117625190___redundant_placeholder2]
Ymodel_price_layer1_while_model_price_layer1_while_cond_117625190___redundant_placeholder3%
!model_price_layer1_while_identity
?
model/price_layer1/while/LessLess$model_price_layer1_while_placeholder@model_price_layer1_while_less_model_price_layer1_strided_slice_1*
T0*
_output_shapes
: 2
model/price_layer1/while/Less?
!model/price_layer1/while/IdentityIdentity!model/price_layer1/while/Less:z:0*
T0
*
_output_shapes
: 2#
!model/price_layer1/while/Identity"O
!model_price_layer1_while_identity*model/price_layer1/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
u
K__inference_concat_layer_layer_call_and_return_conditional_losses_117626271

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
?	
?
)__inference_model_layer_call_fn_117626444
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
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1176264232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::22
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
?
?
0__inference_fixed_layer1_layer_call_fn_117627633

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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_1176262912
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
?
?
while_cond_117627481
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117627481___redundant_placeholder07
3while_while_cond_117627481___redundant_placeholder17
3while_while_cond_117627481___redundant_placeholder27
3while_while_cond_117627481___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?@
?
while_body_117627482
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource??&while/lstm_cell/BiasAdd/ReadVariableOp?%while/lstm_cell/MatMul/ReadVariableOp?'while/lstm_cell/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul?
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp?
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul_1?
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/add?
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp?
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell/split?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul?
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu?
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_1?
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu_1?
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?@
?
while_body_117627329
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_04
0while_lstm_cell_matmul_readvariableop_resource_06
2while_lstm_cell_matmul_1_readvariableop_resource_05
1while_lstm_cell_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor2
.while_lstm_cell_matmul_readvariableop_resource4
0while_lstm_cell_matmul_1_readvariableop_resource3
/while_lstm_cell_biasadd_readvariableop_resource??&while/lstm_cell/BiasAdd/ReadVariableOp?%while/lstm_cell/MatMul/ReadVariableOp?'while/lstm_cell/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
%while/lstm_cell/MatMul/ReadVariableOpReadVariableOp0while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02'
%while/lstm_cell/MatMul/ReadVariableOp?
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0-while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul?
'while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp2while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell/MatMul_1/ReadVariableOp?
while/lstm_cell/MatMul_1MatMulwhile_placeholder_2/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/MatMul_1?
while/lstm_cell/addAddV2 while/lstm_cell/MatMul:product:0"while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/add?
&while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp1while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02(
&while/lstm_cell/BiasAdd/ReadVariableOp?
while/lstm_cell/BiasAddBiasAddwhile/lstm_cell/add:z:0.while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell/BiasAddp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const?
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim?
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0 while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell/split?
while/lstm_cell/SigmoidSigmoidwhile/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid?
while/lstm_cell/Sigmoid_1Sigmoidwhile/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_1?
while/lstm_cell/mulMulwhile/lstm_cell/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul?
while/lstm_cell/ReluReluwhile/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu?
while/lstm_cell/mul_1Mulwhile/lstm_cell/Sigmoid:y:0"while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_1?
while/lstm_cell/add_1AddV2while/lstm_cell/mul:z:0while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/add_1?
while/lstm_cell/Sigmoid_2Sigmoidwhile/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Sigmoid_2?
while/lstm_cell/Relu_1Reluwhile/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/Relu_1?
while/lstm_cell/mul_2Mulwhile/lstm_cell/Sigmoid_2:y:0$while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell/mul_2:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell/add_1:z:0'^while/lstm_cell/BiasAdd/ReadVariableOp&^while/lstm_cell/MatMul/ReadVariableOp(^while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"d
/while_lstm_cell_biasadd_readvariableop_resource1while_lstm_cell_biasadd_readvariableop_resource_0"f
0while_lstm_cell_matmul_1_readvariableop_resource2while_lstm_cell_matmul_1_readvariableop_resource_0"b
.while_lstm_cell_matmul_readvariableop_resource0while_lstm_cell_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2P
&while/lstm_cell/BiasAdd/ReadVariableOp&while/lstm_cell/BiasAdd/ReadVariableOp2N
%while/lstm_cell/MatMul/ReadVariableOp%while/lstm_cell/MatMul/ReadVariableOp2R
'while/lstm_cell/MatMul_1/ReadVariableOp'while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?	
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_117626291

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
?Y
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627567
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117627482* 
condR
while_cond_117627481*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
!price_layer1_while_cond_1176267756
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_38
4price_layer1_while_less_price_layer1_strided_slice_1Q
Mprice_layer1_while_price_layer1_while_cond_117626775___redundant_placeholder0Q
Mprice_layer1_while_price_layer1_while_cond_117626775___redundant_placeholder1Q
Mprice_layer1_while_price_layer1_while_cond_117626775___redundant_placeholder2Q
Mprice_layer1_while_price_layer1_while_cond_117626775___redundant_placeholder3
price_layer1_while_identity
?
price_layer1/while/LessLessprice_layer1_while_placeholder4price_layer1_while_less_price_layer1_strided_slice_1*
T0*
_output_shapes
: 2
price_layer1/while/Less?
price_layer1/while/IdentityIdentityprice_layer1/while/Less:z:0*
T0
*
_output_shapes
: 2
price_layer1/while/Identity"C
price_layer1_while_identity$price_layer1/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
$__inference__wrapped_model_117625300
price_input
	env_input?
;model_price_layer1_lstm_cell_matmul_readvariableop_resourceA
=model_price_layer1_lstm_cell_matmul_1_readvariableop_resource@
<model_price_layer1_lstm_cell_biasadd_readvariableop_resource5
1model_fixed_layer1_matmul_readvariableop_resource6
2model_fixed_layer1_biasadd_readvariableop_resource5
1model_fixed_layer2_matmul_readvariableop_resource6
2model_fixed_layer2_biasadd_readvariableop_resource6
2model_action_output_matmul_readvariableop_resource7
3model_action_output_biasadd_readvariableop_resource
identity??*model/action_output/BiasAdd/ReadVariableOp?)model/action_output/MatMul/ReadVariableOp?)model/fixed_layer1/BiasAdd/ReadVariableOp?(model/fixed_layer1/MatMul/ReadVariableOp?)model/fixed_layer2/BiasAdd/ReadVariableOp?(model/fixed_layer2/MatMul/ReadVariableOp?3model/price_layer1/lstm_cell/BiasAdd/ReadVariableOp?2model/price_layer1/lstm_cell/MatMul/ReadVariableOp?4model/price_layer1/lstm_cell/MatMul_1/ReadVariableOp?model/price_layer1/whileo
model/price_layer1/ShapeShapeprice_input*
T0*
_output_shapes
:2
model/price_layer1/Shape?
&model/price_layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model/price_layer1/strided_slice/stack?
(model/price_layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/price_layer1/strided_slice/stack_1?
(model/price_layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model/price_layer1/strided_slice/stack_2?
 model/price_layer1/strided_sliceStridedSlice!model/price_layer1/Shape:output:0/model/price_layer1/strided_slice/stack:output:01model/price_layer1/strided_slice/stack_1:output:01model/price_layer1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model/price_layer1/strided_slice?
model/price_layer1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2 
model/price_layer1/zeros/mul/y?
model/price_layer1/zeros/mulMul)model/price_layer1/strided_slice:output:0'model/price_layer1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
model/price_layer1/zeros/mul?
model/price_layer1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2!
model/price_layer1/zeros/Less/y?
model/price_layer1/zeros/LessLess model/price_layer1/zeros/mul:z:0(model/price_layer1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
model/price_layer1/zeros/Less?
!model/price_layer1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!model/price_layer1/zeros/packed/1?
model/price_layer1/zeros/packedPack)model/price_layer1/strided_slice:output:0*model/price_layer1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2!
model/price_layer1/zeros/packed?
model/price_layer1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
model/price_layer1/zeros/Const?
model/price_layer1/zerosFill(model/price_layer1/zeros/packed:output:0'model/price_layer1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
model/price_layer1/zeros?
 model/price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/price_layer1/zeros_1/mul/y?
model/price_layer1/zeros_1/mulMul)model/price_layer1/strided_slice:output:0)model/price_layer1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2 
model/price_layer1/zeros_1/mul?
!model/price_layer1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!model/price_layer1/zeros_1/Less/y?
model/price_layer1/zeros_1/LessLess"model/price_layer1/zeros_1/mul:z:0*model/price_layer1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2!
model/price_layer1/zeros_1/Less?
#model/price_layer1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#model/price_layer1/zeros_1/packed/1?
!model/price_layer1/zeros_1/packedPack)model/price_layer1/strided_slice:output:0,model/price_layer1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!model/price_layer1/zeros_1/packed?
 model/price_layer1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model/price_layer1/zeros_1/Const?
model/price_layer1/zeros_1Fill*model/price_layer1/zeros_1/packed:output:0)model/price_layer1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model/price_layer1/zeros_1?
!model/price_layer1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2#
!model/price_layer1/transpose/perm?
model/price_layer1/transpose	Transposeprice_input*model/price_layer1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
model/price_layer1/transpose?
model/price_layer1/Shape_1Shape model/price_layer1/transpose:y:0*
T0*
_output_shapes
:2
model/price_layer1/Shape_1?
(model/price_layer1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/price_layer1/strided_slice_1/stack?
*model/price_layer1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/price_layer1/strided_slice_1/stack_1?
*model/price_layer1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/price_layer1/strided_slice_1/stack_2?
"model/price_layer1/strided_slice_1StridedSlice#model/price_layer1/Shape_1:output:01model/price_layer1/strided_slice_1/stack:output:03model/price_layer1/strided_slice_1/stack_1:output:03model/price_layer1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model/price_layer1/strided_slice_1?
.model/price_layer1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????20
.model/price_layer1/TensorArrayV2/element_shape?
 model/price_layer1/TensorArrayV2TensorListReserve7model/price_layer1/TensorArrayV2/element_shape:output:0+model/price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02"
 model/price_layer1/TensorArrayV2?
Hmodel/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2J
Hmodel/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape?
:model/price_layer1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor model/price_layer1/transpose:y:0Qmodel/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02<
:model/price_layer1/TensorArrayUnstack/TensorListFromTensor?
(model/price_layer1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model/price_layer1/strided_slice_2/stack?
*model/price_layer1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/price_layer1/strided_slice_2/stack_1?
*model/price_layer1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/price_layer1/strided_slice_2/stack_2?
"model/price_layer1/strided_slice_2StridedSlice model/price_layer1/transpose:y:01model/price_layer1/strided_slice_2/stack:output:03model/price_layer1/strided_slice_2/stack_1:output:03model/price_layer1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"model/price_layer1/strided_slice_2?
2model/price_layer1/lstm_cell/MatMul/ReadVariableOpReadVariableOp;model_price_layer1_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype024
2model/price_layer1/lstm_cell/MatMul/ReadVariableOp?
#model/price_layer1/lstm_cell/MatMulMatMul+model/price_layer1/strided_slice_2:output:0:model/price_layer1/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#model/price_layer1/lstm_cell/MatMul?
4model/price_layer1/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp=model_price_layer1_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype026
4model/price_layer1/lstm_cell/MatMul_1/ReadVariableOp?
%model/price_layer1/lstm_cell/MatMul_1MatMul!model/price_layer1/zeros:output:0<model/price_layer1/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%model/price_layer1/lstm_cell/MatMul_1?
 model/price_layer1/lstm_cell/addAddV2-model/price_layer1/lstm_cell/MatMul:product:0/model/price_layer1/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2"
 model/price_layer1/lstm_cell/add?
3model/price_layer1/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp<model_price_layer1_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3model/price_layer1/lstm_cell/BiasAdd/ReadVariableOp?
$model/price_layer1/lstm_cell/BiasAddBiasAdd$model/price_layer1/lstm_cell/add:z:0;model/price_layer1/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$model/price_layer1/lstm_cell/BiasAdd?
"model/price_layer1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"model/price_layer1/lstm_cell/Const?
,model/price_layer1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,model/price_layer1/lstm_cell/split/split_dim?
"model/price_layer1/lstm_cell/splitSplit5model/price_layer1/lstm_cell/split/split_dim:output:0-model/price_layer1/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2$
"model/price_layer1/lstm_cell/split?
$model/price_layer1/lstm_cell/SigmoidSigmoid+model/price_layer1/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2&
$model/price_layer1/lstm_cell/Sigmoid?
&model/price_layer1/lstm_cell/Sigmoid_1Sigmoid+model/price_layer1/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2(
&model/price_layer1/lstm_cell/Sigmoid_1?
 model/price_layer1/lstm_cell/mulMul*model/price_layer1/lstm_cell/Sigmoid_1:y:0#model/price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2"
 model/price_layer1/lstm_cell/mul?
!model/price_layer1/lstm_cell/ReluRelu+model/price_layer1/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2#
!model/price_layer1/lstm_cell/Relu?
"model/price_layer1/lstm_cell/mul_1Mul(model/price_layer1/lstm_cell/Sigmoid:y:0/model/price_layer1/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"model/price_layer1/lstm_cell/mul_1?
"model/price_layer1/lstm_cell/add_1AddV2$model/price_layer1/lstm_cell/mul:z:0&model/price_layer1/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2$
"model/price_layer1/lstm_cell/add_1?
&model/price_layer1/lstm_cell/Sigmoid_2Sigmoid+model/price_layer1/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2(
&model/price_layer1/lstm_cell/Sigmoid_2?
#model/price_layer1/lstm_cell/Relu_1Relu&model/price_layer1/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2%
#model/price_layer1/lstm_cell/Relu_1?
"model/price_layer1/lstm_cell/mul_2Mul*model/price_layer1/lstm_cell/Sigmoid_2:y:01model/price_layer1/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2$
"model/price_layer1/lstm_cell/mul_2?
0model/price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0model/price_layer1/TensorArrayV2_1/element_shape?
"model/price_layer1/TensorArrayV2_1TensorListReserve9model/price_layer1/TensorArrayV2_1/element_shape:output:0+model/price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model/price_layer1/TensorArrayV2_1t
model/price_layer1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model/price_layer1/time?
+model/price_layer1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2-
+model/price_layer1/while/maximum_iterations?
%model/price_layer1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model/price_layer1/while/loop_counter?
model/price_layer1/whileWhile.model/price_layer1/while/loop_counter:output:04model/price_layer1/while/maximum_iterations:output:0 model/price_layer1/time:output:0+model/price_layer1/TensorArrayV2_1:handle:0!model/price_layer1/zeros:output:0#model/price_layer1/zeros_1:output:0+model/price_layer1/strided_slice_1:output:0Jmodel/price_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:0;model_price_layer1_lstm_cell_matmul_readvariableop_resource=model_price_layer1_lstm_cell_matmul_1_readvariableop_resource<model_price_layer1_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*3
body+R)
'model_price_layer1_while_body_117625191*3
cond+R)
'model_price_layer1_while_cond_117625190*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
model/price_layer1/while?
Cmodel/price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2E
Cmodel/price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
5model/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStack!model/price_layer1/while:output:3Lmodel/price_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype027
5model/price_layer1/TensorArrayV2Stack/TensorListStack?
(model/price_layer1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2*
(model/price_layer1/strided_slice_3/stack?
*model/price_layer1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*model/price_layer1/strided_slice_3/stack_1?
*model/price_layer1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model/price_layer1/strided_slice_3/stack_2?
"model/price_layer1/strided_slice_3StridedSlice>model/price_layer1/TensorArrayV2Stack/TensorListStack:tensor:01model/price_layer1/strided_slice_3/stack:output:03model/price_layer1/strided_slice_3/stack_1:output:03model/price_layer1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2$
"model/price_layer1/strided_slice_3?
#model/price_layer1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model/price_layer1/transpose_1/perm?
model/price_layer1/transpose_1	Transpose>model/price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0,model/price_layer1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2 
model/price_layer1/transpose_1?
model/price_layer1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model/price_layer1/runtime?
model/price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/price_flatten/Const?
model/price_flatten/ReshapeReshape+model/price_layer1/strided_slice_3:output:0"model/price_flatten/Const:output:0*
T0*'
_output_shapes
:?????????2
model/price_flatten/Reshape?
model/concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2 
model/concat_layer/concat/axis?
model/concat_layer/concatConcatV2$model/price_flatten/Reshape:output:0	env_input'model/concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????
2
model/concat_layer/concat?
(model/fixed_layer1/MatMul/ReadVariableOpReadVariableOp1model_fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02*
(model/fixed_layer1/MatMul/ReadVariableOp?
model/fixed_layer1/MatMulMatMul"model/concat_layer/concat:output:00model/fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/fixed_layer1/MatMul?
)model/fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp2model_fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model/fixed_layer1/BiasAdd/ReadVariableOp?
model/fixed_layer1/BiasAddBiasAdd#model/fixed_layer1/MatMul:product:01model/fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/fixed_layer1/BiasAdd?
model/fixed_layer1/ReluRelu#model/fixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/fixed_layer1/Relu?
(model/fixed_layer2/MatMul/ReadVariableOpReadVariableOp1model_fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(model/fixed_layer2/MatMul/ReadVariableOp?
model/fixed_layer2/MatMulMatMul%model/fixed_layer1/Relu:activations:00model/fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/fixed_layer2/MatMul?
)model/fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp2model_fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model/fixed_layer2/BiasAdd/ReadVariableOp?
model/fixed_layer2/BiasAddBiasAdd#model/fixed_layer2/MatMul:product:01model/fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/fixed_layer2/BiasAdd?
model/fixed_layer2/ReluRelu#model/fixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model/fixed_layer2/Relu?
)model/action_output/MatMul/ReadVariableOpReadVariableOp2model_action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02+
)model/action_output/MatMul/ReadVariableOp?
model/action_output/MatMulMatMul%model/fixed_layer2/Relu:activations:01model/action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/action_output/MatMul?
*model/action_output/BiasAdd/ReadVariableOpReadVariableOp3model_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*model/action_output/BiasAdd/ReadVariableOp?
model/action_output/BiasAddBiasAdd$model/action_output/MatMul:product:02model/action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/action_output/BiasAdd?
IdentityIdentity$model/action_output/BiasAdd:output:0+^model/action_output/BiasAdd/ReadVariableOp*^model/action_output/MatMul/ReadVariableOp*^model/fixed_layer1/BiasAdd/ReadVariableOp)^model/fixed_layer1/MatMul/ReadVariableOp*^model/fixed_layer2/BiasAdd/ReadVariableOp)^model/fixed_layer2/MatMul/ReadVariableOp4^model/price_layer1/lstm_cell/BiasAdd/ReadVariableOp3^model/price_layer1/lstm_cell/MatMul/ReadVariableOp5^model/price_layer1/lstm_cell/MatMul_1/ReadVariableOp^model/price_layer1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2X
*model/action_output/BiasAdd/ReadVariableOp*model/action_output/BiasAdd/ReadVariableOp2V
)model/action_output/MatMul/ReadVariableOp)model/action_output/MatMul/ReadVariableOp2V
)model/fixed_layer1/BiasAdd/ReadVariableOp)model/fixed_layer1/BiasAdd/ReadVariableOp2T
(model/fixed_layer1/MatMul/ReadVariableOp(model/fixed_layer1/MatMul/ReadVariableOp2V
)model/fixed_layer2/BiasAdd/ReadVariableOp)model/fixed_layer2/BiasAdd/ReadVariableOp2T
(model/fixed_layer2/MatMul/ReadVariableOp(model/fixed_layer2/MatMul/ReadVariableOp2j
3model/price_layer1/lstm_cell/BiasAdd/ReadVariableOp3model/price_layer1/lstm_cell/BiasAdd/ReadVariableOp2h
2model/price_layer1/lstm_cell/MatMul/ReadVariableOp2model/price_layer1/lstm_cell/MatMul/ReadVariableOp2l
4model/price_layer1/lstm_cell/MatMul_1/ReadVariableOp4model/price_layer1/lstm_cell/MatMul_1/ReadVariableOp24
model/price_layer1/whilemodel/price_layer1/while:X T
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
?
?
0__inference_price_layer1_layer_call_fn_117627589
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176259012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
w
K__inference_concat_layer_layer_call_and_return_conditional_losses_117627607
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
?
!price_layer1_while_cond_1176265976
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_38
4price_layer1_while_less_price_layer1_strided_slice_1Q
Mprice_layer1_while_price_layer1_while_cond_117626597___redundant_placeholder0Q
Mprice_layer1_while_price_layer1_while_cond_117626597___redundant_placeholder1Q
Mprice_layer1_while_price_layer1_while_cond_117626597___redundant_placeholder2Q
Mprice_layer1_while_price_layer1_while_cond_117626597___redundant_placeholder3
price_layer1_while_identity
?
price_layer1/while/LessLessprice_layer1_while_placeholder4price_layer1_while_less_price_layer1_strided_slice_1*
T0*
_output_shapes
: 2
price_layer1/while/Less?
price_layer1/while/IdentityIdentityprice_layer1/while/Less:z:0*
T0
*
_output_shapes
: 2
price_layer1/while/Identity"C
price_layer1_while_identity$price_layer1/while/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?
?
-__inference_lstm_cell_layer_call_fn_117627772

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_lstm_cell_layer_call_and_return_conditional_losses_1176254062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:?????????:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
states/1
?]
?
'model_price_layer1_while_body_117625191B
>model_price_layer1_while_model_price_layer1_while_loop_counterH
Dmodel_price_layer1_while_model_price_layer1_while_maximum_iterations(
$model_price_layer1_while_placeholder*
&model_price_layer1_while_placeholder_1*
&model_price_layer1_while_placeholder_2*
&model_price_layer1_while_placeholder_3A
=model_price_layer1_while_model_price_layer1_strided_slice_1_0}
ymodel_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_price_layer1_tensorarrayunstack_tensorlistfromtensor_0G
Cmodel_price_layer1_while_lstm_cell_matmul_readvariableop_resource_0I
Emodel_price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0H
Dmodel_price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0%
!model_price_layer1_while_identity'
#model_price_layer1_while_identity_1'
#model_price_layer1_while_identity_2'
#model_price_layer1_while_identity_3'
#model_price_layer1_while_identity_4'
#model_price_layer1_while_identity_5?
;model_price_layer1_while_model_price_layer1_strided_slice_1{
wmodel_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_price_layer1_tensorarrayunstack_tensorlistfromtensorE
Amodel_price_layer1_while_lstm_cell_matmul_readvariableop_resourceG
Cmodel_price_layer1_while_lstm_cell_matmul_1_readvariableop_resourceF
Bmodel_price_layer1_while_lstm_cell_biasadd_readvariableop_resource??9model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp?8model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp?:model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp?
Jmodel/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jmodel/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
<model/price_layer1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemymodel_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_price_layer1_tensorarrayunstack_tensorlistfromtensor_0$model_price_layer1_while_placeholderSmodel/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02>
<model/price_layer1/while/TensorArrayV2Read/TensorListGetItem?
8model/price_layer1/while/lstm_cell/MatMul/ReadVariableOpReadVariableOpCmodel_price_layer1_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02:
8model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp?
)model/price_layer1/while/lstm_cell/MatMulMatMulCmodel/price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0@model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2+
)model/price_layer1/while/lstm_cell/MatMul?
:model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEmodel_price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02<
:model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp?
+model/price_layer1/while/lstm_cell/MatMul_1MatMul&model_price_layer1_while_placeholder_2Bmodel/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2-
+model/price_layer1/while/lstm_cell/MatMul_1?
&model/price_layer1/while/lstm_cell/addAddV23model/price_layer1/while/lstm_cell/MatMul:product:05model/price_layer1/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2(
&model/price_layer1/while/lstm_cell/add?
9model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDmodel_price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02;
9model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp?
*model/price_layer1/while/lstm_cell/BiasAddBiasAdd*model/price_layer1/while/lstm_cell/add:z:0Amodel/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2,
*model/price_layer1/while/lstm_cell/BiasAdd?
(model/price_layer1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2*
(model/price_layer1/while/lstm_cell/Const?
2model/price_layer1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2model/price_layer1/while/lstm_cell/split/split_dim?
(model/price_layer1/while/lstm_cell/splitSplit;model/price_layer1/while/lstm_cell/split/split_dim:output:03model/price_layer1/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2*
(model/price_layer1/while/lstm_cell/split?
*model/price_layer1/while/lstm_cell/SigmoidSigmoid1model/price_layer1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2,
*model/price_layer1/while/lstm_cell/Sigmoid?
,model/price_layer1/while/lstm_cell/Sigmoid_1Sigmoid1model/price_layer1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2.
,model/price_layer1/while/lstm_cell/Sigmoid_1?
&model/price_layer1/while/lstm_cell/mulMul0model/price_layer1/while/lstm_cell/Sigmoid_1:y:0&model_price_layer1_while_placeholder_3*
T0*'
_output_shapes
:?????????2(
&model/price_layer1/while/lstm_cell/mul?
'model/price_layer1/while/lstm_cell/ReluRelu1model/price_layer1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2)
'model/price_layer1/while/lstm_cell/Relu?
(model/price_layer1/while/lstm_cell/mul_1Mul.model/price_layer1/while/lstm_cell/Sigmoid:y:05model/price_layer1/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2*
(model/price_layer1/while/lstm_cell/mul_1?
(model/price_layer1/while/lstm_cell/add_1AddV2*model/price_layer1/while/lstm_cell/mul:z:0,model/price_layer1/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2*
(model/price_layer1/while/lstm_cell/add_1?
,model/price_layer1/while/lstm_cell/Sigmoid_2Sigmoid1model/price_layer1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2.
,model/price_layer1/while/lstm_cell/Sigmoid_2?
)model/price_layer1/while/lstm_cell/Relu_1Relu,model/price_layer1/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2+
)model/price_layer1/while/lstm_cell/Relu_1?
(model/price_layer1/while/lstm_cell/mul_2Mul0model/price_layer1/while/lstm_cell/Sigmoid_2:y:07model/price_layer1/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2*
(model/price_layer1/while/lstm_cell/mul_2?
=model/price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&model_price_layer1_while_placeholder_1$model_price_layer1_while_placeholder,model/price_layer1/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype02?
=model/price_layer1/while/TensorArrayV2Write/TensorListSetItem?
model/price_layer1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2 
model/price_layer1/while/add/y?
model/price_layer1/while/addAddV2$model_price_layer1_while_placeholder'model/price_layer1/while/add/y:output:0*
T0*
_output_shapes
: 2
model/price_layer1/while/add?
 model/price_layer1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model/price_layer1/while/add_1/y?
model/price_layer1/while/add_1AddV2>model_price_layer1_while_model_price_layer1_while_loop_counter)model/price_layer1/while/add_1/y:output:0*
T0*
_output_shapes
: 2 
model/price_layer1/while/add_1?
!model/price_layer1/while/IdentityIdentity"model/price_layer1/while/add_1:z:0:^model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9^model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp;^model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2#
!model/price_layer1/while/Identity?
#model/price_layer1/while/Identity_1IdentityDmodel_price_layer1_while_model_price_layer1_while_maximum_iterations:^model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9^model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp;^model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model/price_layer1/while/Identity_1?
#model/price_layer1/while/Identity_2Identity model/price_layer1/while/add:z:0:^model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9^model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp;^model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model/price_layer1/while/Identity_2?
#model/price_layer1/while/Identity_3IdentityMmodel/price_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9^model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp;^model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model/price_layer1/while/Identity_3?
#model/price_layer1/while/Identity_4Identity,model/price_layer1/while/lstm_cell/mul_2:z:0:^model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9^model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp;^model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2%
#model/price_layer1/while/Identity_4?
#model/price_layer1/while/Identity_5Identity,model/price_layer1/while/lstm_cell/add_1:z:0:^model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9^model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp;^model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2%
#model/price_layer1/while/Identity_5"O
!model_price_layer1_while_identity*model/price_layer1/while/Identity:output:0"S
#model_price_layer1_while_identity_1,model/price_layer1/while/Identity_1:output:0"S
#model_price_layer1_while_identity_2,model/price_layer1/while/Identity_2:output:0"S
#model_price_layer1_while_identity_3,model/price_layer1/while/Identity_3:output:0"S
#model_price_layer1_while_identity_4,model/price_layer1/while/Identity_4:output:0"S
#model_price_layer1_while_identity_5,model/price_layer1/while/Identity_5:output:0"?
Bmodel_price_layer1_while_lstm_cell_biasadd_readvariableop_resourceDmodel_price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0"?
Cmodel_price_layer1_while_lstm_cell_matmul_1_readvariableop_resourceEmodel_price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0"?
Amodel_price_layer1_while_lstm_cell_matmul_readvariableop_resourceCmodel_price_layer1_while_lstm_cell_matmul_readvariableop_resource_0"|
;model_price_layer1_while_model_price_layer1_strided_slice_1=model_price_layer1_while_model_price_layer1_strided_slice_1_0"?
wmodel_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_price_layer1_tensorarrayunstack_tensorlistfromtensorymodel_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2v
9model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp9model/price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp2t
8model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp8model/price_layer1/while/lstm_cell/MatMul/ReadVariableOp2x
:model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp:model/price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
? 
?
D__inference_model_layer_call_and_return_conditional_losses_117626476

inputs
inputs_1
price_layer1_117626451
price_layer1_117626453
price_layer1_117626455
fixed_layer1_117626460
fixed_layer1_117626462
fixed_layer2_117626465
fixed_layer2_117626467
action_output_117626470
action_output_117626472
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_117626451price_layer1_117626453price_layer1_117626455*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176262202&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_1176262562
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_1176262712
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_117626460fixed_layer1_117626462*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_1176262912&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_117626465fixed_layer2_117626467*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_1176263182&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_117626470action_output_117626472*
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
L__inference_action_output_layer_call_and_return_conditional_losses_1176263442'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
D__inference_model_layer_call_and_return_conditional_losses_117626423

inputs
inputs_1
price_layer1_117626398
price_layer1_117626400
price_layer1_117626402
fixed_layer1_117626407
fixed_layer1_117626409
fixed_layer2_117626412
fixed_layer2_117626414
action_output_117626417
action_output_117626419
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_117626398price_layer1_117626400price_layer1_117626402*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176260672&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_1176262562
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_1176262712
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_117626407fixed_layer1_117626409*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_1176262912&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_117626412fixed_layer2_117626414*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_1176263182&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_117626417action_output_117626419*
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
L__inference_action_output_layer_call_and_return_conditional_losses_1176263442'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_117627328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117627328___redundant_placeholder07
3while_while_cond_117627328___redundant_placeholder17
3while_while_cond_117627328___redundant_placeholder27
3while_while_cond_117627328___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_117627624

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
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_117627663

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
?Y
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117626220

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117626135* 
condR
while_cond_117626134*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
? 
?
D__inference_model_layer_call_and_return_conditional_losses_117626390
price_input
	env_input
price_layer1_117626365
price_layer1_117626367
price_layer1_117626369
fixed_layer1_117626374
fixed_layer1_117626376
fixed_layer2_117626379
fixed_layer2_117626381
action_output_117626384
action_output_117626386
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_117626365price_layer1_117626367price_layer1_117626369*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176262202&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_1176262562
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_1176262712
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_117626374fixed_layer1_117626376*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_1176262912&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_117626379fixed_layer2_117626381*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_1176263182&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_117626384action_output_117626386*
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
L__inference_action_output_layer_call_and_return_conditional_losses_1176263442'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:X T
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
?
?
while_cond_117626134
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117626134___redundant_placeholder07
3while_while_cond_117626134___redundant_placeholder17
3while_while_cond_117626134___redundant_placeholder27
3while_while_cond_117626134___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_117626318

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
?T
?
!price_layer1_while_body_1176265986
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_35
1price_layer1_while_price_layer1_strided_slice_1_0q
mprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0A
=price_layer1_while_lstm_cell_matmul_readvariableop_resource_0C
?price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0B
>price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0
price_layer1_while_identity!
price_layer1_while_identity_1!
price_layer1_while_identity_2!
price_layer1_while_identity_3!
price_layer1_while_identity_4!
price_layer1_while_identity_53
/price_layer1_while_price_layer1_strided_slice_1o
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor?
;price_layer1_while_lstm_cell_matmul_readvariableop_resourceA
=price_layer1_while_lstm_cell_matmul_1_readvariableop_resource@
<price_layer1_while_lstm_cell_biasadd_readvariableop_resource??3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp?2price_layer1/while/lstm_cell/MatMul/ReadVariableOp?4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp?
Dprice_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dprice_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6price_layer1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0price_layer1_while_placeholderMprice_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6price_layer1/while/TensorArrayV2Read/TensorListGetItem?
2price_layer1/while/lstm_cell/MatMul/ReadVariableOpReadVariableOp=price_layer1_while_lstm_cell_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype024
2price_layer1/while/lstm_cell/MatMul/ReadVariableOp?
#price_layer1/while/lstm_cell/MatMulMatMul=price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0:price_layer1/while/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2%
#price_layer1/while/lstm_cell/MatMul?
4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp?price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype026
4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp?
%price_layer1/while/lstm_cell/MatMul_1MatMul price_layer1_while_placeholder_2<price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell/MatMul_1?
 price_layer1/while/lstm_cell/addAddV2-price_layer1/while/lstm_cell/MatMul:product:0/price_layer1/while/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2"
 price_layer1/while/lstm_cell/add?
3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp>price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype025
3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp?
$price_layer1/while/lstm_cell/BiasAddBiasAdd$price_layer1/while/lstm_cell/add:z:0;price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2&
$price_layer1/while/lstm_cell/BiasAdd?
"price_layer1/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2$
"price_layer1/while/lstm_cell/Const?
,price_layer1/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2.
,price_layer1/while/lstm_cell/split/split_dim?
"price_layer1/while/lstm_cell/splitSplit5price_layer1/while/lstm_cell/split/split_dim:output:0-price_layer1/while/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2$
"price_layer1/while/lstm_cell/split?
$price_layer1/while/lstm_cell/SigmoidSigmoid+price_layer1/while/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell/Sigmoid?
&price_layer1/while/lstm_cell/Sigmoid_1Sigmoid+price_layer1/while/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2(
&price_layer1/while/lstm_cell/Sigmoid_1?
 price_layer1/while/lstm_cell/mulMul*price_layer1/while/lstm_cell/Sigmoid_1:y:0 price_layer1_while_placeholder_3*
T0*'
_output_shapes
:?????????2"
 price_layer1/while/lstm_cell/mul?
!price_layer1/while/lstm_cell/ReluRelu+price_layer1/while/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2#
!price_layer1/while/lstm_cell/Relu?
"price_layer1/while/lstm_cell/mul_1Mul(price_layer1/while/lstm_cell/Sigmoid:y:0/price_layer1/while/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell/mul_1?
"price_layer1/while/lstm_cell/add_1AddV2$price_layer1/while/lstm_cell/mul:z:0&price_layer1/while/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell/add_1?
&price_layer1/while/lstm_cell/Sigmoid_2Sigmoid+price_layer1/while/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2(
&price_layer1/while/lstm_cell/Sigmoid_2?
#price_layer1/while/lstm_cell/Relu_1Relu&price_layer1/while/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2%
#price_layer1/while/lstm_cell/Relu_1?
"price_layer1/while/lstm_cell/mul_2Mul*price_layer1/while/lstm_cell/Sigmoid_2:y:01price_layer1/while/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell/mul_2?
7price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer1_while_placeholder_1price_layer1_while_placeholder&price_layer1/while/lstm_cell/mul_2:z:0*
_output_shapes
: *
element_dtype029
7price_layer1/while/TensorArrayV2Write/TensorListSetItemv
price_layer1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/while/add/y?
price_layer1/while/addAddV2price_layer1_while_placeholder!price_layer1/while/add/y:output:0*
T0*
_output_shapes
: 2
price_layer1/while/addz
price_layer1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/while/add_1/y?
price_layer1/while/add_1AddV22price_layer1_while_price_layer1_while_loop_counter#price_layer1/while/add_1/y:output:0*
T0*
_output_shapes
: 2
price_layer1/while/add_1?
price_layer1/while/IdentityIdentityprice_layer1/while/add_1:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity?
price_layer1/while/Identity_1Identity8price_layer1_while_price_layer1_while_maximum_iterations4^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_1?
price_layer1/while/Identity_2Identityprice_layer1/while/add:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_2?
price_layer1/while/Identity_3IdentityGprice_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_3?
price_layer1/while/Identity_4Identity&price_layer1/while/lstm_cell/mul_2:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_4?
price_layer1/while/Identity_5Identity&price_layer1/while/lstm_cell/add_1:z:04^price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3^price_layer1/while/lstm_cell/MatMul/ReadVariableOp5^price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_5"C
price_layer1_while_identity$price_layer1/while/Identity:output:0"G
price_layer1_while_identity_1&price_layer1/while/Identity_1:output:0"G
price_layer1_while_identity_2&price_layer1/while/Identity_2:output:0"G
price_layer1_while_identity_3&price_layer1/while/Identity_3:output:0"G
price_layer1_while_identity_4&price_layer1/while/Identity_4:output:0"G
price_layer1_while_identity_5&price_layer1/while/Identity_5:output:0"~
<price_layer1_while_lstm_cell_biasadd_readvariableop_resource>price_layer1_while_lstm_cell_biasadd_readvariableop_resource_0"?
=price_layer1_while_lstm_cell_matmul_1_readvariableop_resource?price_layer1_while_lstm_cell_matmul_1_readvariableop_resource_0"|
;price_layer1_while_lstm_cell_matmul_readvariableop_resource=price_layer1_while_lstm_cell_matmul_readvariableop_resource_0"d
/price_layer1_while_price_layer1_strided_slice_11price_layer1_while_price_layer1_strided_slice_1_0"?
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensormprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2j
3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp3price_layer1/while/lstm_cell/BiasAdd/ReadVariableOp2h
2price_layer1/while/lstm_cell/MatMul/ReadVariableOp2price_layer1/while/lstm_cell/MatMul/ReadVariableOp2l
4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp4price_layer1/while/lstm_cell/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: 
?Y
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627086

inputs,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:?????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117627001* 
condR
while_cond_117627000*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
)__inference_model_layer_call_fn_117626933
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
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1176264762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::22
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
?	
?
)__inference_model_layer_call_fn_117626497
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
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*+
_read_only_resource_inputs
		
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_model_layer_call_and_return_conditional_losses_1176264762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::22
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
?Y
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627414
inputs_0,
(lstm_cell_matmul_readvariableop_resource.
*lstm_cell_matmul_1_readvariableop_resource-
)lstm_cell_biasadd_readvariableop_resource
identity?? lstm_cell/BiasAdd/ReadVariableOp?lstm_cell/MatMul/ReadVariableOp?!lstm_cell/MatMul_1/ReadVariableOp?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessf
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
zeros_1/packed/1?
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const}
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_2?
lstm_cell/MatMul/ReadVariableOpReadVariableOp(lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02!
lstm_cell/MatMul/ReadVariableOp?
lstm_cell/MatMulMatMulstrided_slice_2:output:0'lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul?
!lstm_cell/MatMul_1/ReadVariableOpReadVariableOp*lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell/MatMul_1/ReadVariableOp?
lstm_cell/MatMul_1MatMulzeros:output:0)lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/MatMul_1?
lstm_cell/addAddV2lstm_cell/MatMul:product:0lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/add?
 lstm_cell/BiasAdd/ReadVariableOpReadVariableOp)lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 lstm_cell/BiasAdd/ReadVariableOp?
lstm_cell/BiasAddBiasAddlstm_cell/add:z:0(lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell/BiasAddd
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dim?
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell/split}
lstm_cell/SigmoidSigmoidlstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid?
lstm_cell/Sigmoid_1Sigmoidlstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_1?
lstm_cell/mulMullstm_cell/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mult
lstm_cell/ReluRelulstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu?
lstm_cell/mul_1Mullstm_cell/Sigmoid:y:0lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_1?
lstm_cell/add_1AddV2lstm_cell/mul:z:0lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/add_1?
lstm_cell/Sigmoid_2Sigmoidlstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell/Sigmoid_2s
lstm_cell/Relu_1Relulstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell/Relu_1?
lstm_cell/mul_2Mullstm_cell/Sigmoid_2:y:0lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0(lstm_cell_matmul_readvariableop_resource*lstm_cell_matmul_1_readvariableop_resource)lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_117627329* 
condR
while_cond_117627328*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^lstm_cell/BiasAdd/ReadVariableOp ^lstm_cell/MatMul/ReadVariableOp"^lstm_cell/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2D
 lstm_cell/BiasAdd/ReadVariableOp lstm_cell/BiasAdd/ReadVariableOp2B
lstm_cell/MatMul/ReadVariableOplstm_cell/MatMul/ReadVariableOp2F
!lstm_cell/MatMul_1/ReadVariableOp!lstm_cell/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_117627000
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117627000___redundant_placeholder07
3while_while_cond_117627000___redundant_placeholder17
3while_while_cond_117627000___redundant_placeholder27
3while_while_cond_117627000___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
?(
?
"__inference__traced_save_117627841
file_prefix2
.savev2_fixed_layer1_kernel_read_readvariableop0
,savev2_fixed_layer1_bias_read_readvariableop2
.savev2_fixed_layer2_kernel_read_readvariableop0
,savev2_fixed_layer2_bias_read_readvariableop3
/savev2_action_output_kernel_read_readvariableop1
-savev2_action_output_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop<
8savev2_price_layer1_lstm_cell_kernel_read_readvariableopF
Bsavev2_price_layer1_lstm_cell_recurrent_kernel_read_readvariableop:
6savev2_price_layer1_lstm_cell_bias_read_readvariableop$
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
:*
dtype0*?
value?B?B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_fixed_layer1_kernel_read_readvariableop,savev2_fixed_layer1_bias_read_readvariableop.savev2_fixed_layer2_kernel_read_readvariableop,savev2_fixed_layer2_bias_read_readvariableop/savev2_action_output_kernel_read_readvariableop-savev2_action_output_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop8savev2_price_layer1_lstm_cell_kernel_read_readvariableopBsavev2_price_layer1_lstm_cell_recurrent_kernel_read_readvariableop6savev2_price_layer1_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*m
_input_shapes\
Z: :
:::::: : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :$ 

_output_shapes

: :$ 

_output_shapes

: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_117625699
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_117625699___redundant_placeholder07
3while_while_cond_117625699___redundant_placeholder17
3while_while_cond_117625699___redundant_placeholder27
3while_while_cond_117625699___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*S
_input_shapesB
@: : : : :?????????:?????????: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????:-)
'
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
:
??
?
D__inference_model_layer_call_and_return_conditional_losses_117626885
inputs_0
inputs_19
5price_layer1_lstm_cell_matmul_readvariableop_resource;
7price_layer1_lstm_cell_matmul_1_readvariableop_resource:
6price_layer1_lstm_cell_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?-price_layer1/lstm_cell/BiasAdd/ReadVariableOp?,price_layer1/lstm_cell/MatMul/ReadVariableOp?.price_layer1/lstm_cell/MatMul_1/ReadVariableOp?price_layer1/while`
price_layer1/ShapeShapeinputs_0*
T0*
_output_shapes
:2
price_layer1/Shape?
 price_layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 price_layer1/strided_slice/stack?
"price_layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer1/strided_slice/stack_1?
"price_layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer1/strided_slice/stack_2?
price_layer1/strided_sliceStridedSliceprice_layer1/Shape:output:0)price_layer1/strided_slice/stack:output:0+price_layer1/strided_slice/stack_1:output:0+price_layer1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer1/strided_slicev
price_layer1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros/mul/y?
price_layer1/zeros/mulMul#price_layer1/strided_slice:output:0!price_layer1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros/muly
price_layer1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer1/zeros/Less/y?
price_layer1/zeros/LessLessprice_layer1/zeros/mul:z:0"price_layer1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros/Less|
price_layer1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros/packed/1?
price_layer1/zeros/packedPack#price_layer1/strided_slice:output:0$price_layer1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer1/zeros/packedy
price_layer1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer1/zeros/Const?
price_layer1/zerosFill"price_layer1/zeros/packed:output:0!price_layer1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/zerosz
price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros_1/mul/y?
price_layer1/zeros_1/mulMul#price_layer1/strided_slice:output:0#price_layer1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros_1/mul}
price_layer1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer1/zeros_1/Less/y?
price_layer1/zeros_1/LessLessprice_layer1/zeros_1/mul:z:0$price_layer1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer1/zeros_1/Less?
price_layer1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/zeros_1/packed/1?
price_layer1/zeros_1/packedPack#price_layer1/strided_slice:output:0&price_layer1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer1/zeros_1/packed}
price_layer1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer1/zeros_1/Const?
price_layer1/zeros_1Fill$price_layer1/zeros_1/packed:output:0#price_layer1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/zeros_1?
price_layer1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer1/transpose/perm?
price_layer1/transpose	Transposeinputs_0$price_layer1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
price_layer1/transposev
price_layer1/Shape_1Shapeprice_layer1/transpose:y:0*
T0*
_output_shapes
:2
price_layer1/Shape_1?
"price_layer1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer1/strided_slice_1/stack?
$price_layer1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_1/stack_1?
$price_layer1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_1/stack_2?
price_layer1/strided_slice_1StridedSliceprice_layer1/Shape_1:output:0+price_layer1/strided_slice_1/stack:output:0-price_layer1/strided_slice_1/stack_1:output:0-price_layer1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer1/strided_slice_1?
(price_layer1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(price_layer1/TensorArrayV2/element_shape?
price_layer1/TensorArrayV2TensorListReserve1price_layer1/TensorArrayV2/element_shape:output:0%price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer1/TensorArrayV2?
Bprice_layer1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bprice_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape?
4price_layer1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorprice_layer1/transpose:y:0Kprice_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4price_layer1/TensorArrayUnstack/TensorListFromTensor?
"price_layer1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer1/strided_slice_2/stack?
$price_layer1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_2/stack_1?
$price_layer1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_2/stack_2?
price_layer1/strided_slice_2StridedSliceprice_layer1/transpose:y:0+price_layer1/strided_slice_2/stack:output:0-price_layer1/strided_slice_2/stack_1:output:0-price_layer1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
price_layer1/strided_slice_2?
,price_layer1/lstm_cell/MatMul/ReadVariableOpReadVariableOp5price_layer1_lstm_cell_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,price_layer1/lstm_cell/MatMul/ReadVariableOp?
price_layer1/lstm_cell/MatMulMatMul%price_layer1/strided_slice_2:output:04price_layer1/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell/MatMul?
.price_layer1/lstm_cell/MatMul_1/ReadVariableOpReadVariableOp7price_layer1_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype020
.price_layer1/lstm_cell/MatMul_1/ReadVariableOp?
price_layer1/lstm_cell/MatMul_1MatMulprice_layer1/zeros:output:06price_layer1/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell/MatMul_1?
price_layer1/lstm_cell/addAddV2'price_layer1/lstm_cell/MatMul:product:0)price_layer1/lstm_cell/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell/add?
-price_layer1/lstm_cell/BiasAdd/ReadVariableOpReadVariableOp6price_layer1_lstm_cell_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-price_layer1/lstm_cell/BiasAdd/ReadVariableOp?
price_layer1/lstm_cell/BiasAddBiasAddprice_layer1/lstm_cell/add:z:05price_layer1/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
price_layer1/lstm_cell/BiasAdd~
price_layer1/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer1/lstm_cell/Const?
&price_layer1/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&price_layer1/lstm_cell/split/split_dim?
price_layer1/lstm_cell/splitSplit/price_layer1/lstm_cell/split/split_dim:output:0'price_layer1/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
price_layer1/lstm_cell/split?
price_layer1/lstm_cell/SigmoidSigmoid%price_layer1/lstm_cell/split:output:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell/Sigmoid?
 price_layer1/lstm_cell/Sigmoid_1Sigmoid%price_layer1/lstm_cell/split:output:1*
T0*'
_output_shapes
:?????????2"
 price_layer1/lstm_cell/Sigmoid_1?
price_layer1/lstm_cell/mulMul$price_layer1/lstm_cell/Sigmoid_1:y:0price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/mul?
price_layer1/lstm_cell/ReluRelu%price_layer1/lstm_cell/split:output:2*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/Relu?
price_layer1/lstm_cell/mul_1Mul"price_layer1/lstm_cell/Sigmoid:y:0)price_layer1/lstm_cell/Relu:activations:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/mul_1?
price_layer1/lstm_cell/add_1AddV2price_layer1/lstm_cell/mul:z:0 price_layer1/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/add_1?
 price_layer1/lstm_cell/Sigmoid_2Sigmoid%price_layer1/lstm_cell/split:output:3*
T0*'
_output_shapes
:?????????2"
 price_layer1/lstm_cell/Sigmoid_2?
price_layer1/lstm_cell/Relu_1Relu price_layer1/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/Relu_1?
price_layer1/lstm_cell/mul_2Mul$price_layer1/lstm_cell/Sigmoid_2:y:0+price_layer1/lstm_cell/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell/mul_2?
*price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2,
*price_layer1/TensorArrayV2_1/element_shape?
price_layer1/TensorArrayV2_1TensorListReserve3price_layer1/TensorArrayV2_1/element_shape:output:0%price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer1/TensorArrayV2_1h
price_layer1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer1/time?
%price_layer1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%price_layer1/while/maximum_iterations?
price_layer1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
price_layer1/while/loop_counter?
price_layer1/whileWhile(price_layer1/while/loop_counter:output:0.price_layer1/while/maximum_iterations:output:0price_layer1/time:output:0%price_layer1/TensorArrayV2_1:handle:0price_layer1/zeros:output:0price_layer1/zeros_1:output:0%price_layer1/strided_slice_1:output:0Dprice_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:05price_layer1_lstm_cell_matmul_readvariableop_resource7price_layer1_lstm_cell_matmul_1_readvariableop_resource6price_layer1_lstm_cell_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!price_layer1_while_body_117626776*-
cond%R#
!price_layer1_while_cond_117626775*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
price_layer1/while?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStackprice_layer1/while:output:3Fprice_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype021
/price_layer1/TensorArrayV2Stack/TensorListStack?
"price_layer1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"price_layer1/strided_slice_3/stack?
$price_layer1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$price_layer1/strided_slice_3/stack_1?
$price_layer1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer1/strided_slice_3/stack_2?
price_layer1/strided_slice_3StridedSlice8price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0+price_layer1/strided_slice_3/stack:output:0-price_layer1/strided_slice_3/stack_1:output:0-price_layer1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
price_layer1/strided_slice_3?
price_layer1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer1/transpose_1/perm?
price_layer1/transpose_1	Transpose8price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0&price_layer1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2
price_layer1/transpose_1?
price_layer1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer1/runtime{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
price_flatten/Const?
price_flatten/ReshapeReshape%price_layer1/strided_slice_3:output:0price_flatten/Const:output:0*
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
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp.^price_layer1/lstm_cell/BiasAdd/ReadVariableOp-^price_layer1/lstm_cell/MatMul/ReadVariableOp/^price_layer1/lstm_cell/MatMul_1/ReadVariableOp^price_layer1/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2^
-price_layer1/lstm_cell/BiasAdd/ReadVariableOp-price_layer1/lstm_cell/BiasAdd/ReadVariableOp2\
,price_layer1/lstm_cell/MatMul/ReadVariableOp,price_layer1/lstm_cell/MatMul/ReadVariableOp2`
.price_layer1/lstm_cell/MatMul_1/ReadVariableOp.price_layer1/lstm_cell/MatMul_1/ReadVariableOp2(
price_layer1/whileprice_layer1/while:U Q
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
? 
?
D__inference_model_layer_call_and_return_conditional_losses_117626361
price_input
	env_input
price_layer1_117626243
price_layer1_117626245
price_layer1_117626247
fixed_layer1_117626302
fixed_layer1_117626304
fixed_layer2_117626329
fixed_layer2_117626331
action_output_117626355
action_output_117626357
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_117626243price_layer1_117626245price_layer1_117626247*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_1176260672&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_1176262562
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_1176262712
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_117626302fixed_layer1_117626304*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_1176262912&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_117626329fixed_layer2_117626331*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_1176263182&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_117626355action_output_117626357*
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
L__inference_action_output_layer_call_and_return_conditional_losses_1176263442'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????:::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input"?L
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
StatefulPartitionedCall:0?????????tensorflow/serving/predict:߇
?A
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		optimizer

loss
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*i&call_and_return_all_conditional_losses
j_default_save_signature
k__call__"?>
_tf_keras_network?>{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5, 1]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}}, "training_config": {"loss": {"action_output": "mse"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "price_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}}
?
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "price_layer1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "price_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "env_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 8]}, {"class_name": "TensorShape", "items": [null, 2]}]}
?

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

*kernel
+bias
,regularization_losses
-	variables
.trainable_variables
/	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "action_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
I
0iter
	1decay
2learning_rate
3momentum"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
_
40
51
62
3
4
$5
%6
*7
+8"
trackable_list_wrapper
_
40
51
62
3
4
$5
%6
*7
+8"
trackable_list_wrapper
?
7metrics
8layer_regularization_losses

9layers
:non_trainable_variables
regularization_losses
	variables
;layer_metrics
trainable_variables
k__call__
j_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
xserving_default"
signature_map
?

4kernel
5recurrent_kernel
6bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
?
@metrics
Alayer_regularization_losses

Blayers
Cnon_trainable_variables
regularization_losses
trainable_variables
	variables
Dlayer_metrics

Estates
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Flayer_regularization_losses
Gmetrics

Hlayers
Inon_trainable_variables
regularization_losses
	variables
Jlayer_metrics
trainable_variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Klayer_regularization_losses
Lmetrics

Mlayers
Nnon_trainable_variables
regularization_losses
	variables
Olayer_metrics
trainable_variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
%:#
2fixed_layer1/kernel
:2fixed_layer1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Player_regularization_losses
Qmetrics

Rlayers
Snon_trainable_variables
 regularization_losses
!	variables
Tlayer_metrics
"trainable_variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
%:#2fixed_layer2/kernel
:2fixed_layer2/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
Ulayer_regularization_losses
Vmetrics

Wlayers
Xnon_trainable_variables
&regularization_losses
'	variables
Ylayer_metrics
(trainable_variables
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
&:$2action_output/kernel
 :2action_output/bias
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
Zlayer_regularization_losses
[metrics

\layers
]non_trainable_variables
,regularization_losses
-	variables
^layer_metrics
.trainable_variables
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
/:- 2price_layer1/lstm_cell/kernel
9:7 2'price_layer1/lstm_cell/recurrent_kernel
):' 2price_layer1/lstm_cell/bias
'
_0"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
?
`layer_regularization_losses
ametrics

blayers
cnon_trainable_variables
<regularization_losses
=	variables
dlayer_metrics
>trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
	etotal
	fcount
g	variables
h	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
?2?
D__inference_model_layer_call_and_return_conditional_losses_117626361
D__inference_model_layer_call_and_return_conditional_losses_117626885
D__inference_model_layer_call_and_return_conditional_losses_117626390
D__inference_model_layer_call_and_return_conditional_losses_117626707?
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
$__inference__wrapped_model_117625300?
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
?2?
)__inference_model_layer_call_fn_117626444
)__inference_model_layer_call_fn_117626933
)__inference_model_layer_call_fn_117626909
)__inference_model_layer_call_fn_117626497?
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
?2?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627567
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627414
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627086
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627239?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
0__inference_price_layer1_layer_call_fn_117627578
0__inference_price_layer1_layer_call_fn_117627589
0__inference_price_layer1_layer_call_fn_117627250
0__inference_price_layer1_layer_call_fn_117627261?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_price_flatten_layer_call_and_return_conditional_losses_117627595?
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
1__inference_price_flatten_layer_call_fn_117627600?
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_117627607?
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
0__inference_concat_layer_layer_call_fn_117627613?
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_117627624?
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
0__inference_fixed_layer1_layer_call_fn_117627633?
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_117627644?
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
0__inference_fixed_layer2_layer_call_fn_117627653?
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
L__inference_action_output_layer_call_and_return_conditional_losses_117627663?
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
1__inference_action_output_layer_call_fn_117627672?
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
'__inference_signature_wrapper_117626529	env_inputprice_input"?
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
 
?2?
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117627705
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117627738?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_lstm_cell_layer_call_fn_117627772
-__inference_lstm_cell_layer_call_fn_117627755?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
$__inference__wrapped_model_117625300?	456$%*+b?_
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
L__inference_action_output_layer_call_and_return_conditional_losses_117627663\*+/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_action_output_layer_call_fn_117627672O*+/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_concat_layer_layer_call_and_return_conditional_losses_117627607?Z?W
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
0__inference_concat_layer_layer_call_fn_117627613vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_117627624\/?,
%?"
 ?
inputs?????????

? "%?"
?
0?????????
? ?
0__inference_fixed_layer1_layer_call_fn_117627633O/?,
%?"
 ?
inputs?????????

? "???????????
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_117627644\$%/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
0__inference_fixed_layer2_layer_call_fn_117627653O$%/?,
%?"
 ?
inputs?????????
? "???????????
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117627705?456??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
H__inference_lstm_cell_layer_call_and_return_conditional_losses_117627738?456??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "s?p
i?f
?
0/0?????????
E?B
?
0/1/0?????????
?
0/1/1?????????
? ?
-__inference_lstm_cell_layer_call_fn_117627755?456??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
-__inference_lstm_cell_layer_call_fn_117627772?456??}
v?s
 ?
inputs?????????
K?H
"?
states/0?????????
"?
states/1?????????
p 
? "c?`
?
0?????????
A?>
?
1/0?????????
?
1/1??????????
D__inference_model_layer_call_and_return_conditional_losses_117626361?	456$%*+j?g
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
D__inference_model_layer_call_and_return_conditional_losses_117626390?	456$%*+j?g
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
D__inference_model_layer_call_and_return_conditional_losses_117626707?	456$%*+f?c
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
D__inference_model_layer_call_and_return_conditional_losses_117626885?	456$%*+f?c
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
)__inference_model_layer_call_fn_117626444?	456$%*+j?g
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
)__inference_model_layer_call_fn_117626497?	456$%*+j?g
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
)__inference_model_layer_call_fn_117626909?	456$%*+f?c
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
)__inference_model_layer_call_fn_117626933?	456$%*+f?c
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_117627595X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_price_flatten_layer_call_fn_117627600K/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627086m456??<
5?2
$?!
inputs?????????

 
p

 
? "%?"
?
0?????????
? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627239m456??<
5?2
$?!
inputs?????????

 
p 

 
? "%?"
?
0?????????
? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627414}456O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0?????????
? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_117627567}456O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0?????????
? ?
0__inference_price_layer1_layer_call_fn_117627250`456??<
5?2
$?!
inputs?????????

 
p

 
? "???????????
0__inference_price_layer1_layer_call_fn_117627261`456??<
5?2
$?!
inputs?????????

 
p 

 
? "???????????
0__inference_price_layer1_layer_call_fn_117627578p456O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "???????????
0__inference_price_layer1_layer_call_fn_117627589p456O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "???????????
'__inference_signature_wrapper_117626529?	456$%*+y?v
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