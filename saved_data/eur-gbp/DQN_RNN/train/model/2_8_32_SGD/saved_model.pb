??(
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
?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??%
?
fixed_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"*$
shared_namefixed_layer1/kernel
{
'fixed_layer1/kernel/Read/ReadVariableOpReadVariableOpfixed_layer1/kernel*
_output_shapes

:"*
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
price_layer1/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!price_layer1/lstm_cell_4/kernel
?
3price_layer1/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOpprice_layer1/lstm_cell_4/kernel*
_output_shapes

: *
dtype0
?
)price_layer1/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *:
shared_name+)price_layer1/lstm_cell_4/recurrent_kernel
?
=price_layer1/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp)price_layer1/lstm_cell_4/recurrent_kernel*
_output_shapes

: *
dtype0
?
price_layer1/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameprice_layer1/lstm_cell_4/bias
?
1price_layer1/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOpprice_layer1/lstm_cell_4/bias*
_output_shapes
: *
dtype0
?
price_layer2/lstm_cell_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*0
shared_name!price_layer2/lstm_cell_5/kernel
?
3price_layer2/lstm_cell_5/kernel/Read/ReadVariableOpReadVariableOpprice_layer2/lstm_cell_5/kernel*
_output_shapes
:	?*
dtype0
?
)price_layer2/lstm_cell_5/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*:
shared_name+)price_layer2/lstm_cell_5/recurrent_kernel
?
=price_layer2/lstm_cell_5/recurrent_kernel/Read/ReadVariableOpReadVariableOp)price_layer2/lstm_cell_5/recurrent_kernel*
_output_shapes
:	 ?*
dtype0
?
price_layer2/lstm_cell_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*.
shared_nameprice_layer2/lstm_cell_5/bias
?
1price_layer2/lstm_cell_5/bias/Read/ReadVariableOpReadVariableOpprice_layer2/lstm_cell_5/bias*
_output_shapes	
:?*
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
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?+
value?+B?+ B?+
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
loss
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
l
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
	variables
trainable_variables
 	keras_api
 
R
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
h

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
6
7iter
	8decay
9learning_rate
:momentum
 
 
V
;0
<1
=2
>3
?4
@5
%6
&7
+8
,9
110
211
V
;0
<1
=2
>3
?4
@5
%6
&7
+8
,9
110
211
?
Ametrics
Blayer_regularization_losses

Clayers
Dnon_trainable_variables
regularization_losses
	variables
Elayer_metrics
trainable_variables
 
~

;kernel
<recurrent_kernel
=bias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
 
 

;0
<1
=2

;0
<1
=2
?
Jmetrics
Klayer_regularization_losses

Llayers
Mnon_trainable_variables
regularization_losses
trainable_variables
	variables
Nlayer_metrics

Ostates
~

>kernel
?recurrent_kernel
@bias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
 
 

>0
?1
@2

>0
?1
@2
?
Tmetrics
Ulayer_regularization_losses

Vlayers
Wnon_trainable_variables
regularization_losses
trainable_variables
	variables
Xlayer_metrics

Ystates
 
 
 
?
Zlayer_regularization_losses
[metrics

\layers
]non_trainable_variables
regularization_losses
	variables
^layer_metrics
trainable_variables
 
 
 
?
_layer_regularization_losses
`metrics

alayers
bnon_trainable_variables
!regularization_losses
"	variables
clayer_metrics
#trainable_variables
_]
VARIABLE_VALUEfixed_layer1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
?
dlayer_regularization_losses
emetrics

flayers
gnon_trainable_variables
'regularization_losses
(	variables
hlayer_metrics
)trainable_variables
_]
VARIABLE_VALUEfixed_layer2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
ilayer_regularization_losses
jmetrics

klayers
lnon_trainable_variables
-regularization_losses
.	variables
mlayer_metrics
/trainable_variables
`^
VARIABLE_VALUEaction_output/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEaction_output/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
?
nlayer_regularization_losses
ometrics

players
qnon_trainable_variables
3regularization_losses
4	variables
rlayer_metrics
5trainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer1/lstm_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)price_layer1/lstm_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEprice_layer1/lstm_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer2/lstm_cell_5/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)price_layer2/lstm_cell_5/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEprice_layer2/lstm_cell_5/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

s0
 
?
0
1
2
3
4
5
6
7
	8
 
 
 

;0
<1
=2

;0
<1
=2
?
tlayer_regularization_losses
umetrics

vlayers
wnon_trainable_variables
Fregularization_losses
G	variables
xlayer_metrics
Htrainable_variables
 
 

0
 
 
 
 

>0
?1
@2

>0
?1
@2
?
ylayer_regularization_losses
zmetrics

{layers
|non_trainable_variables
Pregularization_losses
Q	variables
}layer_metrics
Rtrainable_variables
 
 

0
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
6
	~total
	count
?	variables
?	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

~0
1

?	variables
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
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_env_inputserving_default_price_inputprice_layer1/lstm_cell_4/kernel)price_layer1/lstm_cell_4/recurrent_kernelprice_layer1/lstm_cell_4/biasprice_layer2/lstm_cell_5/kernel)price_layer2/lstm_cell_5/recurrent_kernelprice_layer2/lstm_cell_5/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_595874300
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'fixed_layer1/kernel/Read/ReadVariableOp%fixed_layer1/bias/Read/ReadVariableOp'fixed_layer2/kernel/Read/ReadVariableOp%fixed_layer2/bias/Read/ReadVariableOp(action_output/kernel/Read/ReadVariableOp&action_output/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp3price_layer1/lstm_cell_4/kernel/Read/ReadVariableOp=price_layer1/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp1price_layer1/lstm_cell_4/bias/Read/ReadVariableOp3price_layer2/lstm_cell_5/kernel/Read/ReadVariableOp=price_layer2/lstm_cell_5/recurrent_kernel/Read/ReadVariableOp1price_layer2/lstm_cell_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
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
"__inference__traced_save_595876687
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumprice_layer1/lstm_cell_4/kernel)price_layer1/lstm_cell_4/recurrent_kernelprice_layer1/lstm_cell_4/biasprice_layer2/lstm_cell_5/kernel)price_layer2/lstm_cell_5/recurrent_kernelprice_layer2/lstm_cell_5/biastotalcount*
Tin
2*
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
%__inference__traced_restore_595876751??$
?a
?
)model_4_price_layer2_while_body_595871978F
Bmodel_4_price_layer2_while_model_4_price_layer2_while_loop_counterL
Hmodel_4_price_layer2_while_model_4_price_layer2_while_maximum_iterations*
&model_4_price_layer2_while_placeholder,
(model_4_price_layer2_while_placeholder_1,
(model_4_price_layer2_while_placeholder_2,
(model_4_price_layer2_while_placeholder_3E
Amodel_4_price_layer2_while_model_4_price_layer2_strided_slice_1_0?
}model_4_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer2_tensorarrayunstack_tensorlistfromtensor_0K
Gmodel_4_price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0M
Imodel_4_price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0L
Hmodel_4_price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0'
#model_4_price_layer2_while_identity)
%model_4_price_layer2_while_identity_1)
%model_4_price_layer2_while_identity_2)
%model_4_price_layer2_while_identity_3)
%model_4_price_layer2_while_identity_4)
%model_4_price_layer2_while_identity_5C
?model_4_price_layer2_while_model_4_price_layer2_strided_slice_1
{model_4_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer2_tensorarrayunstack_tensorlistfromtensorI
Emodel_4_price_layer2_while_lstm_cell_5_matmul_readvariableop_resourceK
Gmodel_4_price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resourceJ
Fmodel_4_price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource??=model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp?<model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?>model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp?
Lmodel_4/price_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lmodel_4/price_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>model_4/price_layer2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_4_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer2_tensorarrayunstack_tensorlistfromtensor_0&model_4_price_layer2_while_placeholderUmodel_4/price_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>model_4/price_layer2/while/TensorArrayV2Read/TensorListGetItem?
<model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOpGmodel_4_price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02>
<model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?
-model_4/price_layer2/while/lstm_cell_5/MatMulMatMulEmodel_4/price_layer2/while/TensorArrayV2Read/TensorListGetItem:item:0Dmodel_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2/
-model_4/price_layer2/while/lstm_cell_5/MatMul?
>model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpImodel_4_price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02@
>model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp?
/model_4/price_layer2/while/lstm_cell_5/MatMul_1MatMul(model_4_price_layer2_while_placeholder_2Fmodel_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/model_4/price_layer2/while/lstm_cell_5/MatMul_1?
*model_4/price_layer2/while/lstm_cell_5/addAddV27model_4/price_layer2/while/lstm_cell_5/MatMul:product:09model_4/price_layer2/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2,
*model_4/price_layer2/while/lstm_cell_5/add?
=model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOpHmodel_4_price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02?
=model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp?
.model_4/price_layer2/while/lstm_cell_5/BiasAddBiasAdd.model_4/price_layer2/while/lstm_cell_5/add:z:0Emodel_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.model_4/price_layer2/while/lstm_cell_5/BiasAdd?
,model_4/price_layer2/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_4/price_layer2/while/lstm_cell_5/Const?
6model_4/price_layer2/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6model_4/price_layer2/while/lstm_cell_5/split/split_dim?
,model_4/price_layer2/while/lstm_cell_5/splitSplit?model_4/price_layer2/while/lstm_cell_5/split/split_dim:output:07model_4/price_layer2/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2.
,model_4/price_layer2/while/lstm_cell_5/split?
.model_4/price_layer2/while/lstm_cell_5/SigmoidSigmoid5model_4/price_layer2/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 20
.model_4/price_layer2/while/lstm_cell_5/Sigmoid?
0model_4/price_layer2/while/lstm_cell_5/Sigmoid_1Sigmoid5model_4/price_layer2/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 22
0model_4/price_layer2/while/lstm_cell_5/Sigmoid_1?
*model_4/price_layer2/while/lstm_cell_5/mulMul4model_4/price_layer2/while/lstm_cell_5/Sigmoid_1:y:0(model_4_price_layer2_while_placeholder_3*
T0*'
_output_shapes
:????????? 2,
*model_4/price_layer2/while/lstm_cell_5/mul?
+model_4/price_layer2/while/lstm_cell_5/ReluRelu5model_4/price_layer2/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2-
+model_4/price_layer2/while/lstm_cell_5/Relu?
,model_4/price_layer2/while/lstm_cell_5/mul_1Mul2model_4/price_layer2/while/lstm_cell_5/Sigmoid:y:09model_4/price_layer2/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2.
,model_4/price_layer2/while/lstm_cell_5/mul_1?
,model_4/price_layer2/while/lstm_cell_5/add_1AddV2.model_4/price_layer2/while/lstm_cell_5/mul:z:00model_4/price_layer2/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2.
,model_4/price_layer2/while/lstm_cell_5/add_1?
0model_4/price_layer2/while/lstm_cell_5/Sigmoid_2Sigmoid5model_4/price_layer2/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 22
0model_4/price_layer2/while/lstm_cell_5/Sigmoid_2?
-model_4/price_layer2/while/lstm_cell_5/Relu_1Relu0model_4/price_layer2/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2/
-model_4/price_layer2/while/lstm_cell_5/Relu_1?
,model_4/price_layer2/while/lstm_cell_5/mul_2Mul4model_4/price_layer2/while/lstm_cell_5/Sigmoid_2:y:0;model_4/price_layer2/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2.
,model_4/price_layer2/while/lstm_cell_5/mul_2?
?model_4/price_layer2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_4_price_layer2_while_placeholder_1&model_4_price_layer2_while_placeholder0model_4/price_layer2/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?model_4/price_layer2/while/TensorArrayV2Write/TensorListSetItem?
 model_4/price_layer2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_4/price_layer2/while/add/y?
model_4/price_layer2/while/addAddV2&model_4_price_layer2_while_placeholder)model_4/price_layer2/while/add/y:output:0*
T0*
_output_shapes
: 2 
model_4/price_layer2/while/add?
"model_4/price_layer2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_4/price_layer2/while/add_1/y?
 model_4/price_layer2/while/add_1AddV2Bmodel_4_price_layer2_while_model_4_price_layer2_while_loop_counter+model_4/price_layer2/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 model_4/price_layer2/while/add_1?
#model_4/price_layer2/while/IdentityIdentity$model_4/price_layer2/while/add_1:z:0>^model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=^model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?^model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model_4/price_layer2/while/Identity?
%model_4/price_layer2/while/Identity_1IdentityHmodel_4_price_layer2_while_model_4_price_layer2_while_maximum_iterations>^model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=^model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?^model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_4/price_layer2/while/Identity_1?
%model_4/price_layer2/while/Identity_2Identity"model_4/price_layer2/while/add:z:0>^model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=^model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?^model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_4/price_layer2/while/Identity_2?
%model_4/price_layer2/while/Identity_3IdentityOmodel_4/price_layer2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=^model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?^model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_4/price_layer2/while/Identity_3?
%model_4/price_layer2/while/Identity_4Identity0model_4/price_layer2/while/lstm_cell_5/mul_2:z:0>^model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=^model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?^model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2'
%model_4/price_layer2/while/Identity_4?
%model_4/price_layer2/while/Identity_5Identity0model_4/price_layer2/while/lstm_cell_5/add_1:z:0>^model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=^model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?^model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2'
%model_4/price_layer2/while/Identity_5"S
#model_4_price_layer2_while_identity,model_4/price_layer2/while/Identity:output:0"W
%model_4_price_layer2_while_identity_1.model_4/price_layer2/while/Identity_1:output:0"W
%model_4_price_layer2_while_identity_2.model_4/price_layer2/while/Identity_2:output:0"W
%model_4_price_layer2_while_identity_3.model_4/price_layer2/while/Identity_3:output:0"W
%model_4_price_layer2_while_identity_4.model_4/price_layer2/while/Identity_4:output:0"W
%model_4_price_layer2_while_identity_5.model_4/price_layer2/while/Identity_5:output:0"?
Fmodel_4_price_layer2_while_lstm_cell_5_biasadd_readvariableop_resourceHmodel_4_price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
Gmodel_4_price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resourceImodel_4_price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
Emodel_4_price_layer2_while_lstm_cell_5_matmul_readvariableop_resourceGmodel_4_price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0"?
?model_4_price_layer2_while_model_4_price_layer2_strided_slice_1Amodel_4_price_layer2_while_model_4_price_layer2_strided_slice_1_0"?
{model_4_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer2_tensorarrayunstack_tensorlistfromtensor}model_4_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2~
=model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp=model_4/price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp2|
<model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp<model_4/price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp2?
>model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp>model_4/price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_595872486
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595872486___redundant_placeholder07
3while_while_cond_595872486___redundant_placeholder17
3while_while_cond_595872486___redundant_placeholder27
3while_while_cond_595872486___redundant_placeholder3
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
?.
?
"__inference__traced_save_595876687
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
'savev2_sgd_momentum_read_readvariableop>
:savev2_price_layer1_lstm_cell_4_kernel_read_readvariableopH
Dsavev2_price_layer1_lstm_cell_4_recurrent_kernel_read_readvariableop<
8savev2_price_layer1_lstm_cell_4_bias_read_readvariableop>
:savev2_price_layer2_lstm_cell_5_kernel_read_readvariableopH
Dsavev2_price_layer2_lstm_cell_5_recurrent_kernel_read_readvariableop<
8savev2_price_layer2_lstm_cell_5_bias_read_readvariableop$
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_fixed_layer1_kernel_read_readvariableop,savev2_fixed_layer1_bias_read_readvariableop.savev2_fixed_layer2_kernel_read_readvariableop,savev2_fixed_layer2_bias_read_readvariableop/savev2_action_output_kernel_read_readvariableop-savev2_action_output_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop:savev2_price_layer1_lstm_cell_4_kernel_read_readvariableopDsavev2_price_layer1_lstm_cell_4_recurrent_kernel_read_readvariableop8savev2_price_layer1_lstm_cell_4_bias_read_readvariableop:savev2_price_layer2_lstm_cell_5_kernel_read_readvariableopDsavev2_price_layer2_lstm_cell_5_recurrent_kernel_read_readvariableop8savev2_price_layer2_lstm_cell_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapesy
w: :":::::: : : : : : : :	?:	 ?:?: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:": 
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
: :%!

_output_shapes
:	?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
$__inference__wrapped_model_595872087
price_input
	env_inputC
?model_4_price_layer1_lstm_cell_4_matmul_readvariableop_resourceE
Amodel_4_price_layer1_lstm_cell_4_matmul_1_readvariableop_resourceD
@model_4_price_layer1_lstm_cell_4_biasadd_readvariableop_resourceC
?model_4_price_layer2_lstm_cell_5_matmul_readvariableop_resourceE
Amodel_4_price_layer2_lstm_cell_5_matmul_1_readvariableop_resourceD
@model_4_price_layer2_lstm_cell_5_biasadd_readvariableop_resource7
3model_4_fixed_layer1_matmul_readvariableop_resource8
4model_4_fixed_layer1_biasadd_readvariableop_resource7
3model_4_fixed_layer2_matmul_readvariableop_resource8
4model_4_fixed_layer2_biasadd_readvariableop_resource8
4model_4_action_output_matmul_readvariableop_resource9
5model_4_action_output_biasadd_readvariableop_resource
identity??,model_4/action_output/BiasAdd/ReadVariableOp?+model_4/action_output/MatMul/ReadVariableOp?+model_4/fixed_layer1/BiasAdd/ReadVariableOp?*model_4/fixed_layer1/MatMul/ReadVariableOp?+model_4/fixed_layer2/BiasAdd/ReadVariableOp?*model_4/fixed_layer2/MatMul/ReadVariableOp?7model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp?6model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOp?8model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp?model_4/price_layer1/while?7model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp?6model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOp?8model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp?model_4/price_layer2/whiles
model_4/price_layer1/ShapeShapeprice_input*
T0*
_output_shapes
:2
model_4/price_layer1/Shape?
(model_4/price_layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model_4/price_layer1/strided_slice/stack?
*model_4/price_layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_4/price_layer1/strided_slice/stack_1?
*model_4/price_layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_4/price_layer1/strided_slice/stack_2?
"model_4/price_layer1/strided_sliceStridedSlice#model_4/price_layer1/Shape:output:01model_4/price_layer1/strided_slice/stack:output:03model_4/price_layer1/strided_slice/stack_1:output:03model_4/price_layer1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model_4/price_layer1/strided_slice?
 model_4/price_layer1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_4/price_layer1/zeros/mul/y?
model_4/price_layer1/zeros/mulMul+model_4/price_layer1/strided_slice:output:0)model_4/price_layer1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
model_4/price_layer1/zeros/mul?
!model_4/price_layer1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!model_4/price_layer1/zeros/Less/y?
model_4/price_layer1/zeros/LessLess"model_4/price_layer1/zeros/mul:z:0*model_4/price_layer1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
model_4/price_layer1/zeros/Less?
#model_4/price_layer1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2%
#model_4/price_layer1/zeros/packed/1?
!model_4/price_layer1/zeros/packedPack+model_4/price_layer1/strided_slice:output:0,model_4/price_layer1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!model_4/price_layer1/zeros/packed?
 model_4/price_layer1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_4/price_layer1/zeros/Const?
model_4/price_layer1/zerosFill*model_4/price_layer1/zeros/packed:output:0)model_4/price_layer1/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
model_4/price_layer1/zeros?
"model_4/price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_4/price_layer1/zeros_1/mul/y?
 model_4/price_layer1/zeros_1/mulMul+model_4/price_layer1/strided_slice:output:0+model_4/price_layer1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 model_4/price_layer1/zeros_1/mul?
#model_4/price_layer1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_4/price_layer1/zeros_1/Less/y?
!model_4/price_layer1/zeros_1/LessLess$model_4/price_layer1/zeros_1/mul:z:0,model_4/price_layer1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!model_4/price_layer1/zeros_1/Less?
%model_4/price_layer1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%model_4/price_layer1/zeros_1/packed/1?
#model_4/price_layer1/zeros_1/packedPack+model_4/price_layer1/strided_slice:output:0.model_4/price_layer1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#model_4/price_layer1/zeros_1/packed?
"model_4/price_layer1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_4/price_layer1/zeros_1/Const?
model_4/price_layer1/zeros_1Fill,model_4/price_layer1/zeros_1/packed:output:0+model_4/price_layer1/zeros_1/Const:output:0*
T0*'
_output_shapes
:?????????2
model_4/price_layer1/zeros_1?
#model_4/price_layer1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model_4/price_layer1/transpose/perm?
model_4/price_layer1/transpose	Transposeprice_input,model_4/price_layer1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2 
model_4/price_layer1/transpose?
model_4/price_layer1/Shape_1Shape"model_4/price_layer1/transpose:y:0*
T0*
_output_shapes
:2
model_4/price_layer1/Shape_1?
*model_4/price_layer1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_4/price_layer1/strided_slice_1/stack?
,model_4/price_layer1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer1/strided_slice_1/stack_1?
,model_4/price_layer1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer1/strided_slice_1/stack_2?
$model_4/price_layer1/strided_slice_1StridedSlice%model_4/price_layer1/Shape_1:output:03model_4/price_layer1/strided_slice_1/stack:output:05model_4/price_layer1/strided_slice_1/stack_1:output:05model_4/price_layer1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model_4/price_layer1/strided_slice_1?
0model_4/price_layer1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model_4/price_layer1/TensorArrayV2/element_shape?
"model_4/price_layer1/TensorArrayV2TensorListReserve9model_4/price_layer1/TensorArrayV2/element_shape:output:0-model_4/price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model_4/price_layer1/TensorArrayV2?
Jmodel_4/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jmodel_4/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape?
<model_4/price_layer1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_4/price_layer1/transpose:y:0Smodel_4/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<model_4/price_layer1/TensorArrayUnstack/TensorListFromTensor?
*model_4/price_layer1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_4/price_layer1/strided_slice_2/stack?
,model_4/price_layer1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer1/strided_slice_2/stack_1?
,model_4/price_layer1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer1/strided_slice_2/stack_2?
$model_4/price_layer1/strided_slice_2StridedSlice"model_4/price_layer1/transpose:y:03model_4/price_layer1/strided_slice_2/stack:output:05model_4/price_layer1/strided_slice_2/stack_1:output:05model_4/price_layer1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$model_4/price_layer1/strided_slice_2?
6model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp?model_4_price_layer1_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype028
6model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOp?
'model_4/price_layer1/lstm_cell_4/MatMulMatMul-model_4/price_layer1/strided_slice_2:output:0>model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'model_4/price_layer1/lstm_cell_4/MatMul?
8model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpAmodel_4_price_layer1_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02:
8model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp?
)model_4/price_layer1/lstm_cell_4/MatMul_1MatMul#model_4/price_layer1/zeros:output:0@model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2+
)model_4/price_layer1/lstm_cell_4/MatMul_1?
$model_4/price_layer1/lstm_cell_4/addAddV21model_4/price_layer1/lstm_cell_4/MatMul:product:03model_4/price_layer1/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2&
$model_4/price_layer1/lstm_cell_4/add?
7model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp@model_4_price_layer1_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp?
(model_4/price_layer1/lstm_cell_4/BiasAddBiasAdd(model_4/price_layer1/lstm_cell_4/add:z:0?model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2*
(model_4/price_layer1/lstm_cell_4/BiasAdd?
&model_4/price_layer1/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_4/price_layer1/lstm_cell_4/Const?
0model_4/price_layer1/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_4/price_layer1/lstm_cell_4/split/split_dim?
&model_4/price_layer1/lstm_cell_4/splitSplit9model_4/price_layer1/lstm_cell_4/split/split_dim:output:01model_4/price_layer1/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2(
&model_4/price_layer1/lstm_cell_4/split?
(model_4/price_layer1/lstm_cell_4/SigmoidSigmoid/model_4/price_layer1/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2*
(model_4/price_layer1/lstm_cell_4/Sigmoid?
*model_4/price_layer1/lstm_cell_4/Sigmoid_1Sigmoid/model_4/price_layer1/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2,
*model_4/price_layer1/lstm_cell_4/Sigmoid_1?
$model_4/price_layer1/lstm_cell_4/mulMul.model_4/price_layer1/lstm_cell_4/Sigmoid_1:y:0%model_4/price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2&
$model_4/price_layer1/lstm_cell_4/mul?
%model_4/price_layer1/lstm_cell_4/ReluRelu/model_4/price_layer1/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2'
%model_4/price_layer1/lstm_cell_4/Relu?
&model_4/price_layer1/lstm_cell_4/mul_1Mul,model_4/price_layer1/lstm_cell_4/Sigmoid:y:03model_4/price_layer1/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2(
&model_4/price_layer1/lstm_cell_4/mul_1?
&model_4/price_layer1/lstm_cell_4/add_1AddV2(model_4/price_layer1/lstm_cell_4/mul:z:0*model_4/price_layer1/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2(
&model_4/price_layer1/lstm_cell_4/add_1?
*model_4/price_layer1/lstm_cell_4/Sigmoid_2Sigmoid/model_4/price_layer1/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2,
*model_4/price_layer1/lstm_cell_4/Sigmoid_2?
'model_4/price_layer1/lstm_cell_4/Relu_1Relu*model_4/price_layer1/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2)
'model_4/price_layer1/lstm_cell_4/Relu_1?
&model_4/price_layer1/lstm_cell_4/mul_2Mul.model_4/price_layer1/lstm_cell_4/Sigmoid_2:y:05model_4/price_layer1/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2(
&model_4/price_layer1/lstm_cell_4/mul_2?
2model_4/price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   24
2model_4/price_layer1/TensorArrayV2_1/element_shape?
$model_4/price_layer1/TensorArrayV2_1TensorListReserve;model_4/price_layer1/TensorArrayV2_1/element_shape:output:0-model_4/price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$model_4/price_layer1/TensorArrayV2_1x
model_4/price_layer1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_4/price_layer1/time?
-model_4/price_layer1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model_4/price_layer1/while/maximum_iterations?
'model_4/price_layer1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_4/price_layer1/while/loop_counter?
model_4/price_layer1/whileWhile0model_4/price_layer1/while/loop_counter:output:06model_4/price_layer1/while/maximum_iterations:output:0"model_4/price_layer1/time:output:0-model_4/price_layer1/TensorArrayV2_1:handle:0#model_4/price_layer1/zeros:output:0%model_4/price_layer1/zeros_1:output:0-model_4/price_layer1/strided_slice_1:output:0Lmodel_4/price_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:0?model_4_price_layer1_lstm_cell_4_matmul_readvariableop_resourceAmodel_4_price_layer1_lstm_cell_4_matmul_1_readvariableop_resource@model_4_price_layer1_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :?????????:?????????: : : : : *%
_read_only_resource_inputs
	
*5
body-R+
)model_4_price_layer1_while_body_595871829*5
cond-R+
)model_4_price_layer1_while_cond_595871828*K
output_shapes:
8: : : : :?????????:?????????: : : : : *
parallel_iterations 2
model_4/price_layer1/while?
Emodel_4/price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2G
Emodel_4/price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
7model_4/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStack#model_4/price_layer1/while:output:3Nmodel_4/price_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:?????????*
element_dtype029
7model_4/price_layer1/TensorArrayV2Stack/TensorListStack?
*model_4/price_layer1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*model_4/price_layer1/strided_slice_3/stack?
,model_4/price_layer1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,model_4/price_layer1/strided_slice_3/stack_1?
,model_4/price_layer1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer1/strided_slice_3/stack_2?
$model_4/price_layer1/strided_slice_3StridedSlice@model_4/price_layer1/TensorArrayV2Stack/TensorListStack:tensor:03model_4/price_layer1/strided_slice_3/stack:output:05model_4/price_layer1/strided_slice_3/stack_1:output:05model_4/price_layer1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$model_4/price_layer1/strided_slice_3?
%model_4/price_layer1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%model_4/price_layer1/transpose_1/perm?
 model_4/price_layer1/transpose_1	Transpose@model_4/price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0.model_4/price_layer1/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????2"
 model_4/price_layer1/transpose_1?
model_4/price_layer1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_4/price_layer1/runtime?
model_4/price_layer2/ShapeShape$model_4/price_layer1/transpose_1:y:0*
T0*
_output_shapes
:2
model_4/price_layer2/Shape?
(model_4/price_layer2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model_4/price_layer2/strided_slice/stack?
*model_4/price_layer2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_4/price_layer2/strided_slice/stack_1?
*model_4/price_layer2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_4/price_layer2/strided_slice/stack_2?
"model_4/price_layer2/strided_sliceStridedSlice#model_4/price_layer2/Shape:output:01model_4/price_layer2/strided_slice/stack:output:03model_4/price_layer2/strided_slice/stack_1:output:03model_4/price_layer2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model_4/price_layer2/strided_slice?
 model_4/price_layer2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model_4/price_layer2/zeros/mul/y?
model_4/price_layer2/zeros/mulMul+model_4/price_layer2/strided_slice:output:0)model_4/price_layer2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
model_4/price_layer2/zeros/mul?
!model_4/price_layer2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!model_4/price_layer2/zeros/Less/y?
model_4/price_layer2/zeros/LessLess"model_4/price_layer2/zeros/mul:z:0*model_4/price_layer2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
model_4/price_layer2/zeros/Less?
#model_4/price_layer2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_4/price_layer2/zeros/packed/1?
!model_4/price_layer2/zeros/packedPack+model_4/price_layer2/strided_slice:output:0,model_4/price_layer2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!model_4/price_layer2/zeros/packed?
 model_4/price_layer2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_4/price_layer2/zeros/Const?
model_4/price_layer2/zerosFill*model_4/price_layer2/zeros/packed:output:0)model_4/price_layer2/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_4/price_layer2/zeros?
"model_4/price_layer2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_4/price_layer2/zeros_1/mul/y?
 model_4/price_layer2/zeros_1/mulMul+model_4/price_layer2/strided_slice:output:0+model_4/price_layer2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 model_4/price_layer2/zeros_1/mul?
#model_4/price_layer2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_4/price_layer2/zeros_1/Less/y?
!model_4/price_layer2/zeros_1/LessLess$model_4/price_layer2/zeros_1/mul:z:0,model_4/price_layer2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!model_4/price_layer2/zeros_1/Less?
%model_4/price_layer2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%model_4/price_layer2/zeros_1/packed/1?
#model_4/price_layer2/zeros_1/packedPack+model_4/price_layer2/strided_slice:output:0.model_4/price_layer2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#model_4/price_layer2/zeros_1/packed?
"model_4/price_layer2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_4/price_layer2/zeros_1/Const?
model_4/price_layer2/zeros_1Fill,model_4/price_layer2/zeros_1/packed:output:0+model_4/price_layer2/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_4/price_layer2/zeros_1?
#model_4/price_layer2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model_4/price_layer2/transpose/perm?
model_4/price_layer2/transpose	Transpose$model_4/price_layer1/transpose_1:y:0,model_4/price_layer2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2 
model_4/price_layer2/transpose?
model_4/price_layer2/Shape_1Shape"model_4/price_layer2/transpose:y:0*
T0*
_output_shapes
:2
model_4/price_layer2/Shape_1?
*model_4/price_layer2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_4/price_layer2/strided_slice_1/stack?
,model_4/price_layer2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer2/strided_slice_1/stack_1?
,model_4/price_layer2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer2/strided_slice_1/stack_2?
$model_4/price_layer2/strided_slice_1StridedSlice%model_4/price_layer2/Shape_1:output:03model_4/price_layer2/strided_slice_1/stack:output:05model_4/price_layer2/strided_slice_1/stack_1:output:05model_4/price_layer2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model_4/price_layer2/strided_slice_1?
0model_4/price_layer2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model_4/price_layer2/TensorArrayV2/element_shape?
"model_4/price_layer2/TensorArrayV2TensorListReserve9model_4/price_layer2/TensorArrayV2/element_shape:output:0-model_4/price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model_4/price_layer2/TensorArrayV2?
Jmodel_4/price_layer2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jmodel_4/price_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape?
<model_4/price_layer2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_4/price_layer2/transpose:y:0Smodel_4/price_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<model_4/price_layer2/TensorArrayUnstack/TensorListFromTensor?
*model_4/price_layer2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_4/price_layer2/strided_slice_2/stack?
,model_4/price_layer2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer2/strided_slice_2/stack_1?
,model_4/price_layer2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer2/strided_slice_2/stack_2?
$model_4/price_layer2/strided_slice_2StridedSlice"model_4/price_layer2/transpose:y:03model_4/price_layer2/strided_slice_2/stack:output:05model_4/price_layer2/strided_slice_2/stack_1:output:05model_4/price_layer2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$model_4/price_layer2/strided_slice_2?
6model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp?model_4_price_layer2_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOp?
'model_4/price_layer2/lstm_cell_5/MatMulMatMul-model_4/price_layer2/strided_slice_2:output:0>model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'model_4/price_layer2/lstm_cell_5/MatMul?
8model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpAmodel_4_price_layer2_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02:
8model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp?
)model_4/price_layer2/lstm_cell_5/MatMul_1MatMul#model_4/price_layer2/zeros:output:0@model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_4/price_layer2/lstm_cell_5/MatMul_1?
$model_4/price_layer2/lstm_cell_5/addAddV21model_4/price_layer2/lstm_cell_5/MatMul:product:03model_4/price_layer2/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2&
$model_4/price_layer2/lstm_cell_5/add?
7model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp@model_4_price_layer2_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp?
(model_4/price_layer2/lstm_cell_5/BiasAddBiasAdd(model_4/price_layer2/lstm_cell_5/add:z:0?model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_4/price_layer2/lstm_cell_5/BiasAdd?
&model_4/price_layer2/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2(
&model_4/price_layer2/lstm_cell_5/Const?
0model_4/price_layer2/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :22
0model_4/price_layer2/lstm_cell_5/split/split_dim?
&model_4/price_layer2/lstm_cell_5/splitSplit9model_4/price_layer2/lstm_cell_5/split/split_dim:output:01model_4/price_layer2/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2(
&model_4/price_layer2/lstm_cell_5/split?
(model_4/price_layer2/lstm_cell_5/SigmoidSigmoid/model_4/price_layer2/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2*
(model_4/price_layer2/lstm_cell_5/Sigmoid?
*model_4/price_layer2/lstm_cell_5/Sigmoid_1Sigmoid/model_4/price_layer2/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2,
*model_4/price_layer2/lstm_cell_5/Sigmoid_1?
$model_4/price_layer2/lstm_cell_5/mulMul.model_4/price_layer2/lstm_cell_5/Sigmoid_1:y:0%model_4/price_layer2/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2&
$model_4/price_layer2/lstm_cell_5/mul?
%model_4/price_layer2/lstm_cell_5/ReluRelu/model_4/price_layer2/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2'
%model_4/price_layer2/lstm_cell_5/Relu?
&model_4/price_layer2/lstm_cell_5/mul_1Mul,model_4/price_layer2/lstm_cell_5/Sigmoid:y:03model_4/price_layer2/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2(
&model_4/price_layer2/lstm_cell_5/mul_1?
&model_4/price_layer2/lstm_cell_5/add_1AddV2(model_4/price_layer2/lstm_cell_5/mul:z:0*model_4/price_layer2/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2(
&model_4/price_layer2/lstm_cell_5/add_1?
*model_4/price_layer2/lstm_cell_5/Sigmoid_2Sigmoid/model_4/price_layer2/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2,
*model_4/price_layer2/lstm_cell_5/Sigmoid_2?
'model_4/price_layer2/lstm_cell_5/Relu_1Relu*model_4/price_layer2/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2)
'model_4/price_layer2/lstm_cell_5/Relu_1?
&model_4/price_layer2/lstm_cell_5/mul_2Mul.model_4/price_layer2/lstm_cell_5/Sigmoid_2:y:05model_4/price_layer2/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2(
&model_4/price_layer2/lstm_cell_5/mul_2?
2model_4/price_layer2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    24
2model_4/price_layer2/TensorArrayV2_1/element_shape?
$model_4/price_layer2/TensorArrayV2_1TensorListReserve;model_4/price_layer2/TensorArrayV2_1/element_shape:output:0-model_4/price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$model_4/price_layer2/TensorArrayV2_1x
model_4/price_layer2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_4/price_layer2/time?
-model_4/price_layer2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model_4/price_layer2/while/maximum_iterations?
'model_4/price_layer2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_4/price_layer2/while/loop_counter?
model_4/price_layer2/whileWhile0model_4/price_layer2/while/loop_counter:output:06model_4/price_layer2/while/maximum_iterations:output:0"model_4/price_layer2/time:output:0-model_4/price_layer2/TensorArrayV2_1:handle:0#model_4/price_layer2/zeros:output:0%model_4/price_layer2/zeros_1:output:0-model_4/price_layer2/strided_slice_1:output:0Lmodel_4/price_layer2/TensorArrayUnstack/TensorListFromTensor:output_handle:0?model_4_price_layer2_lstm_cell_5_matmul_readvariableop_resourceAmodel_4_price_layer2_lstm_cell_5_matmul_1_readvariableop_resource@model_4_price_layer2_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*5
body-R+
)model_4_price_layer2_while_body_595871978*5
cond-R+
)model_4_price_layer2_while_cond_595871977*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
model_4/price_layer2/while?
Emodel_4/price_layer2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2G
Emodel_4/price_layer2/TensorArrayV2Stack/TensorListStack/element_shape?
7model_4/price_layer2/TensorArrayV2Stack/TensorListStackTensorListStack#model_4/price_layer2/while:output:3Nmodel_4/price_layer2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype029
7model_4/price_layer2/TensorArrayV2Stack/TensorListStack?
*model_4/price_layer2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*model_4/price_layer2/strided_slice_3/stack?
,model_4/price_layer2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,model_4/price_layer2/strided_slice_3/stack_1?
,model_4/price_layer2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_4/price_layer2/strided_slice_3/stack_2?
$model_4/price_layer2/strided_slice_3StridedSlice@model_4/price_layer2/TensorArrayV2Stack/TensorListStack:tensor:03model_4/price_layer2/strided_slice_3/stack:output:05model_4/price_layer2/strided_slice_3/stack_1:output:05model_4/price_layer2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2&
$model_4/price_layer2/strided_slice_3?
%model_4/price_layer2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%model_4/price_layer2/transpose_1/perm?
 model_4/price_layer2/transpose_1	Transpose@model_4/price_layer2/TensorArrayV2Stack/TensorListStack:tensor:0.model_4/price_layer2/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2"
 model_4/price_layer2/transpose_1?
model_4/price_layer2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_4/price_layer2/runtime?
model_4/price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model_4/price_flatten/Const?
model_4/price_flatten/ReshapeReshape-model_4/price_layer2/strided_slice_3:output:0$model_4/price_flatten/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_4/price_flatten/Reshape?
 model_4/concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_4/concat_layer/concat/axis?
model_4/concat_layer/concatConcatV2&model_4/price_flatten/Reshape:output:0	env_input)model_4/concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????"2
model_4/concat_layer/concat?
*model_4/fixed_layer1/MatMul/ReadVariableOpReadVariableOp3model_4_fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:"*
dtype02,
*model_4/fixed_layer1/MatMul/ReadVariableOp?
model_4/fixed_layer1/MatMulMatMul$model_4/concat_layer/concat:output:02model_4/fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/fixed_layer1/MatMul?
+model_4/fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_4_fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_4/fixed_layer1/BiasAdd/ReadVariableOp?
model_4/fixed_layer1/BiasAddBiasAdd%model_4/fixed_layer1/MatMul:product:03model_4/fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/fixed_layer1/BiasAdd?
model_4/fixed_layer1/ReluRelu%model_4/fixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/fixed_layer1/Relu?
*model_4/fixed_layer2/MatMul/ReadVariableOpReadVariableOp3model_4_fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*model_4/fixed_layer2/MatMul/ReadVariableOp?
model_4/fixed_layer2/MatMulMatMul'model_4/fixed_layer1/Relu:activations:02model_4/fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/fixed_layer2/MatMul?
+model_4/fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_4_fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_4/fixed_layer2/BiasAdd/ReadVariableOp?
model_4/fixed_layer2/BiasAddBiasAdd%model_4/fixed_layer2/MatMul:product:03model_4/fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/fixed_layer2/BiasAdd?
model_4/fixed_layer2/ReluRelu%model_4/fixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_4/fixed_layer2/Relu?
+model_4/action_output/MatMul/ReadVariableOpReadVariableOp4model_4_action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+model_4/action_output/MatMul/ReadVariableOp?
model_4/action_output/MatMulMatMul'model_4/fixed_layer2/Relu:activations:03model_4/action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/action_output/MatMul?
,model_4/action_output/BiasAdd/ReadVariableOpReadVariableOp5model_4_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_4/action_output/BiasAdd/ReadVariableOp?
model_4/action_output/BiasAddBiasAdd&model_4/action_output/MatMul:product:04model_4/action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_4/action_output/BiasAdd?
IdentityIdentity&model_4/action_output/BiasAdd:output:0-^model_4/action_output/BiasAdd/ReadVariableOp,^model_4/action_output/MatMul/ReadVariableOp,^model_4/fixed_layer1/BiasAdd/ReadVariableOp+^model_4/fixed_layer1/MatMul/ReadVariableOp,^model_4/fixed_layer2/BiasAdd/ReadVariableOp+^model_4/fixed_layer2/MatMul/ReadVariableOp8^model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp7^model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOp9^model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp^model_4/price_layer1/while8^model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp7^model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOp9^model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp^model_4/price_layer2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2\
,model_4/action_output/BiasAdd/ReadVariableOp,model_4/action_output/BiasAdd/ReadVariableOp2Z
+model_4/action_output/MatMul/ReadVariableOp+model_4/action_output/MatMul/ReadVariableOp2Z
+model_4/fixed_layer1/BiasAdd/ReadVariableOp+model_4/fixed_layer1/BiasAdd/ReadVariableOp2X
*model_4/fixed_layer1/MatMul/ReadVariableOp*model_4/fixed_layer1/MatMul/ReadVariableOp2Z
+model_4/fixed_layer2/BiasAdd/ReadVariableOp+model_4/fixed_layer2/BiasAdd/ReadVariableOp2X
*model_4/fixed_layer2/MatMul/ReadVariableOp*model_4/fixed_layer2/MatMul/ReadVariableOp2r
7model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp7model_4/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp2p
6model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOp6model_4/price_layer1/lstm_cell_4/MatMul/ReadVariableOp2t
8model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp8model_4/price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp28
model_4/price_layer1/whilemodel_4/price_layer1/while2r
7model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp7model_4/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp2p
6model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOp6model_4/price_layer2/lstm_cell_5/MatMul/ReadVariableOp2t
8model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp8model_4/price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp28
model_4/price_layer2/whilemodel_4/price_layer2/while:X T
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
/__inference_lstm_cell_5_layer_call_fn_595876609

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
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_5958728032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?D
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595872556

inputs
lstm_cell_4_595872474
lstm_cell_4_595872476
lstm_cell_4_595872478
identity??#lstm_cell_4/StatefulPartitionedCall?whileD
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
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_595872474lstm_cell_4_595872476lstm_cell_4_595872478*
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
GPU 2J 8? *S
fNRL
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_5958721602%
#lstm_cell_4/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_595872474lstm_cell_4_595872476lstm_cell_4_595872478*
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
while_body_595872487* 
condR
while_cond_595872486*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
1__inference_action_output_layer_call_fn_595876409

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
L__inference_action_output_layer_call_and_return_conditional_losses_5958740762
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
?
/__inference_lstm_cell_5_layer_call_fn_595876592

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
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_5958727702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?B
?
while_body_595873714
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_595874050

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
?B
?
while_body_595876219
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
/__inference_lstm_cell_4_layer_call_fn_595876509

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
GPU 2J 8? *S
fNRL
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_5958721932
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
?
?
!price_layer1_while_cond_5958743686
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_38
4price_layer1_while_less_price_layer1_strided_slice_1Q
Mprice_layer1_while_price_layer1_while_cond_595874368___redundant_placeholder0Q
Mprice_layer1_while_price_layer1_while_cond_595874368___redundant_placeholder1Q
Mprice_layer1_while_price_layer1_while_cond_595874368___redundant_placeholder2Q
Mprice_layer1_while_price_layer1_while_cond_595874368___redundant_placeholder3
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_595876361

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:"*
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
:?????????"::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?
?
/__inference_lstm_cell_4_layer_call_fn_595876492

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
GPU 2J 8? *S
fNRL
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_5958721602
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
while_cond_595875890
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595875890___redundant_placeholder07
3while_while_cond_595875890___redundant_placeholder17
3while_while_cond_595875890___redundant_placeholder27
3while_while_cond_595875890___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?

?
+__inference_model_4_layer_call_fn_595874196
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
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_5958741692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::22
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
?
?
while_cond_595872618
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595872618___redundant_placeholder07
3while_while_cond_595872618___redundant_placeholder17
3while_while_cond_595872618___redundant_placeholder27
3while_while_cond_595872618___redundant_placeholder3
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
?
?
while_cond_595873713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595873713___redundant_placeholder07
3while_while_cond_595873713___redundant_placeholder17
3while_while_cond_595873713___redundant_placeholder27
3while_while_cond_595873713___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?D
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595873166

inputs
lstm_cell_5_595873084
lstm_cell_5_595873086
lstm_cell_5_595873088
identity??#lstm_cell_5/StatefulPartitionedCall?whileD
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_595873084lstm_cell_5_595873086lstm_cell_5_595873088*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_5958727702%
#lstm_cell_5/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_595873084lstm_cell_5_595873086lstm_cell_5_595873088*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595873097* 
condR
while_cond_595873096*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
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
:????????? *
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
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595872770

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
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?
?
while_cond_595873866
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595873866___redundant_placeholder07
3while_while_cond_595873866___redundant_placeholder17
3while_while_cond_595873866___redundant_placeholder27
3while_while_cond_595873866___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
while_cond_595873228
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595873228___redundant_placeholder07
3while_while_cond_595873228___redundant_placeholder17
3while_while_cond_595873228___redundant_placeholder27
3while_while_cond_595873228___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
while_cond_595873378
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595873378___redundant_placeholder07
3while_while_cond_595873378___redundant_placeholder17
3while_while_cond_595873378___redundant_placeholder27
3while_while_cond_595873378___redundant_placeholder3
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
?
?
0__inference_fixed_layer1_layer_call_fn_595876370

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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_5958740232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????"::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?Z
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875648

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??"lstm_cell_4/BiasAdd/ReadVariableOp?!lstm_cell_4/MatMul/ReadVariableOp?#lstm_cell_4/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
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
while_body_595875563* 
condR
while_cond_595875562*K
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
IdentityIdentitytranspose_1:y:0#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_signature_wrapper_595874300
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
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_5958720872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::22
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
?B
?
while_body_595875235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??(while/lstm_cell_4/BiasAdd/ReadVariableOp?'while/lstm_cell_4/MatMul/ReadVariableOp?)while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595872803

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
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?[
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875320
inputs_0.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??"lstm_cell_4/BiasAdd/ReadVariableOp?!lstm_cell_4/MatMul/ReadVariableOp?#lstm_cell_4/MatMul_1/ReadVariableOp?whileF
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
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
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
while_body_595875235* 
condR
while_cond_595875234*K
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
IdentityIdentitytranspose_1:y:0#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
0__inference_price_layer2_layer_call_fn_595875987
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
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958731662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
0__inference_price_layer2_layer_call_fn_595876315

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
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958737992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
while_cond_595875409
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595875409___redundant_placeholder07
3while_while_cond_595875409___redundant_placeholder17
3while_while_cond_595875409___redundant_placeholder27
3while_while_cond_595875409___redundant_placeholder3
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
?
?
while_cond_595875562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595875562___redundant_placeholder07
3while_while_cond_595875562___redundant_placeholder17
3while_while_cond_595875562___redundant_placeholder27
3while_while_cond_595875562___redundant_placeholder3
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
?a
?
)model_4_price_layer1_while_body_595871829F
Bmodel_4_price_layer1_while_model_4_price_layer1_while_loop_counterL
Hmodel_4_price_layer1_while_model_4_price_layer1_while_maximum_iterations*
&model_4_price_layer1_while_placeholder,
(model_4_price_layer1_while_placeholder_1,
(model_4_price_layer1_while_placeholder_2,
(model_4_price_layer1_while_placeholder_3E
Amodel_4_price_layer1_while_model_4_price_layer1_strided_slice_1_0?
}model_4_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer1_tensorarrayunstack_tensorlistfromtensor_0K
Gmodel_4_price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0M
Imodel_4_price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0L
Hmodel_4_price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0'
#model_4_price_layer1_while_identity)
%model_4_price_layer1_while_identity_1)
%model_4_price_layer1_while_identity_2)
%model_4_price_layer1_while_identity_3)
%model_4_price_layer1_while_identity_4)
%model_4_price_layer1_while_identity_5C
?model_4_price_layer1_while_model_4_price_layer1_strided_slice_1
{model_4_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer1_tensorarrayunstack_tensorlistfromtensorI
Emodel_4_price_layer1_while_lstm_cell_4_matmul_readvariableop_resourceK
Gmodel_4_price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resourceJ
Fmodel_4_price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource??=model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp?<model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?>model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp?
Lmodel_4/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lmodel_4/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>model_4/price_layer1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_4_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer1_tensorarrayunstack_tensorlistfromtensor_0&model_4_price_layer1_while_placeholderUmodel_4/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>model_4/price_layer1/while/TensorArrayV2Read/TensorListGetItem?
<model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpGmodel_4_price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02>
<model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?
-model_4/price_layer1/while/lstm_cell_4/MatMulMatMulEmodel_4/price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0Dmodel_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2/
-model_4/price_layer1/while/lstm_cell_4/MatMul?
>model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpImodel_4_price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02@
>model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp?
/model_4/price_layer1/while/lstm_cell_4/MatMul_1MatMul(model_4_price_layer1_while_placeholder_2Fmodel_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 21
/model_4/price_layer1/while/lstm_cell_4/MatMul_1?
*model_4/price_layer1/while/lstm_cell_4/addAddV27model_4/price_layer1/while/lstm_cell_4/MatMul:product:09model_4/price_layer1/while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2,
*model_4/price_layer1/while/lstm_cell_4/add?
=model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOpHmodel_4_price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02?
=model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp?
.model_4/price_layer1/while/lstm_cell_4/BiasAddBiasAdd.model_4/price_layer1/while/lstm_cell_4/add:z:0Emodel_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 20
.model_4/price_layer1/while/lstm_cell_4/BiasAdd?
,model_4/price_layer1/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_4/price_layer1/while/lstm_cell_4/Const?
6model_4/price_layer1/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :28
6model_4/price_layer1/while/lstm_cell_4/split/split_dim?
,model_4/price_layer1/while/lstm_cell_4/splitSplit?model_4/price_layer1/while/lstm_cell_4/split/split_dim:output:07model_4/price_layer1/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2.
,model_4/price_layer1/while/lstm_cell_4/split?
.model_4/price_layer1/while/lstm_cell_4/SigmoidSigmoid5model_4/price_layer1/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????20
.model_4/price_layer1/while/lstm_cell_4/Sigmoid?
0model_4/price_layer1/while/lstm_cell_4/Sigmoid_1Sigmoid5model_4/price_layer1/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????22
0model_4/price_layer1/while/lstm_cell_4/Sigmoid_1?
*model_4/price_layer1/while/lstm_cell_4/mulMul4model_4/price_layer1/while/lstm_cell_4/Sigmoid_1:y:0(model_4_price_layer1_while_placeholder_3*
T0*'
_output_shapes
:?????????2,
*model_4/price_layer1/while/lstm_cell_4/mul?
+model_4/price_layer1/while/lstm_cell_4/ReluRelu5model_4/price_layer1/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2-
+model_4/price_layer1/while/lstm_cell_4/Relu?
,model_4/price_layer1/while/lstm_cell_4/mul_1Mul2model_4/price_layer1/while/lstm_cell_4/Sigmoid:y:09model_4/price_layer1/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2.
,model_4/price_layer1/while/lstm_cell_4/mul_1?
,model_4/price_layer1/while/lstm_cell_4/add_1AddV2.model_4/price_layer1/while/lstm_cell_4/mul:z:00model_4/price_layer1/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2.
,model_4/price_layer1/while/lstm_cell_4/add_1?
0model_4/price_layer1/while/lstm_cell_4/Sigmoid_2Sigmoid5model_4/price_layer1/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????22
0model_4/price_layer1/while/lstm_cell_4/Sigmoid_2?
-model_4/price_layer1/while/lstm_cell_4/Relu_1Relu0model_4/price_layer1/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2/
-model_4/price_layer1/while/lstm_cell_4/Relu_1?
,model_4/price_layer1/while/lstm_cell_4/mul_2Mul4model_4/price_layer1/while/lstm_cell_4/Sigmoid_2:y:0;model_4/price_layer1/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2.
,model_4/price_layer1/while/lstm_cell_4/mul_2?
?model_4/price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_4_price_layer1_while_placeholder_1&model_4_price_layer1_while_placeholder0model_4/price_layer1/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?model_4/price_layer1/while/TensorArrayV2Write/TensorListSetItem?
 model_4/price_layer1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_4/price_layer1/while/add/y?
model_4/price_layer1/while/addAddV2&model_4_price_layer1_while_placeholder)model_4/price_layer1/while/add/y:output:0*
T0*
_output_shapes
: 2 
model_4/price_layer1/while/add?
"model_4/price_layer1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_4/price_layer1/while/add_1/y?
 model_4/price_layer1/while/add_1AddV2Bmodel_4_price_layer1_while_model_4_price_layer1_while_loop_counter+model_4/price_layer1/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 model_4/price_layer1/while/add_1?
#model_4/price_layer1/while/IdentityIdentity$model_4/price_layer1/while/add_1:z:0>^model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=^model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?^model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model_4/price_layer1/while/Identity?
%model_4/price_layer1/while/Identity_1IdentityHmodel_4_price_layer1_while_model_4_price_layer1_while_maximum_iterations>^model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=^model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?^model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_4/price_layer1/while/Identity_1?
%model_4/price_layer1/while/Identity_2Identity"model_4/price_layer1/while/add:z:0>^model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=^model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?^model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_4/price_layer1/while/Identity_2?
%model_4/price_layer1/while/Identity_3IdentityOmodel_4/price_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0>^model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=^model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?^model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_4/price_layer1/while/Identity_3?
%model_4/price_layer1/while/Identity_4Identity0model_4/price_layer1/while/lstm_cell_4/mul_2:z:0>^model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=^model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?^model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2'
%model_4/price_layer1/while/Identity_4?
%model_4/price_layer1/while/Identity_5Identity0model_4/price_layer1/while/lstm_cell_4/add_1:z:0>^model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=^model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?^model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2'
%model_4/price_layer1/while/Identity_5"S
#model_4_price_layer1_while_identity,model_4/price_layer1/while/Identity:output:0"W
%model_4_price_layer1_while_identity_1.model_4/price_layer1/while/Identity_1:output:0"W
%model_4_price_layer1_while_identity_2.model_4/price_layer1/while/Identity_2:output:0"W
%model_4_price_layer1_while_identity_3.model_4/price_layer1/while/Identity_3:output:0"W
%model_4_price_layer1_while_identity_4.model_4/price_layer1/while/Identity_4:output:0"W
%model_4_price_layer1_while_identity_5.model_4/price_layer1/while/Identity_5:output:0"?
Fmodel_4_price_layer1_while_lstm_cell_4_biasadd_readvariableop_resourceHmodel_4_price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0"?
Gmodel_4_price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resourceImodel_4_price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0"?
Emodel_4_price_layer1_while_lstm_cell_4_matmul_readvariableop_resourceGmodel_4_price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0"?
?model_4_price_layer1_while_model_4_price_layer1_strided_slice_1Amodel_4_price_layer1_while_model_4_price_layer1_strided_slice_1_0"?
{model_4_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer1_tensorarrayunstack_tensorlistfromtensor}model_4_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_4_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2~
=model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp=model_4/price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp2|
<model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp<model_4/price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp2?
>model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp>model_4/price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
?
\
0__inference_concat_layer_layer_call_fn_595876350
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
:?????????"* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_5958740032
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?U
?
!price_layer1_while_body_5958746966
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_35
1price_layer1_while_price_layer1_strided_slice_1_0q
mprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0C
?price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0E
Aprice_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0D
@price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0
price_layer1_while_identity!
price_layer1_while_identity_1!
price_layer1_while_identity_2!
price_layer1_while_identity_3!
price_layer1_while_identity_4!
price_layer1_while_identity_53
/price_layer1_while_price_layer1_strided_slice_1o
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensorA
=price_layer1_while_lstm_cell_4_matmul_readvariableop_resourceC
?price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resourceB
>price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource??5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp?4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp?price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype026
4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?
%price_layer1/while/lstm_cell_4/MatMulMatMul=price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0<price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_4/MatMul?
6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpAprice_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype028
6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp?
'price_layer1/while/lstm_cell_4/MatMul_1MatMul price_layer1_while_placeholder_2>price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'price_layer1/while/lstm_cell_4/MatMul_1?
"price_layer1/while/lstm_cell_4/addAddV2/price_layer1/while/lstm_cell_4/MatMul:product:01price_layer1/while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2$
"price_layer1/while/lstm_cell_4/add?
5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp@price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype027
5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp?
&price_layer1/while/lstm_cell_4/BiasAddBiasAdd&price_layer1/while/lstm_cell_4/add:z:0=price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&price_layer1/while/lstm_cell_4/BiasAdd?
$price_layer1/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$price_layer1/while/lstm_cell_4/Const?
.price_layer1/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.price_layer1/while/lstm_cell_4/split/split_dim?
$price_layer1/while/lstm_cell_4/splitSplit7price_layer1/while/lstm_cell_4/split/split_dim:output:0/price_layer1/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2&
$price_layer1/while/lstm_cell_4/split?
&price_layer1/while/lstm_cell_4/SigmoidSigmoid-price_layer1/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2(
&price_layer1/while/lstm_cell_4/Sigmoid?
(price_layer1/while/lstm_cell_4/Sigmoid_1Sigmoid-price_layer1/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2*
(price_layer1/while/lstm_cell_4/Sigmoid_1?
"price_layer1/while/lstm_cell_4/mulMul,price_layer1/while/lstm_cell_4/Sigmoid_1:y:0 price_layer1_while_placeholder_3*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell_4/mul?
#price_layer1/while/lstm_cell_4/ReluRelu-price_layer1/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2%
#price_layer1/while/lstm_cell_4/Relu?
$price_layer1/while/lstm_cell_4/mul_1Mul*price_layer1/while/lstm_cell_4/Sigmoid:y:01price_layer1/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell_4/mul_1?
$price_layer1/while/lstm_cell_4/add_1AddV2&price_layer1/while/lstm_cell_4/mul:z:0(price_layer1/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell_4/add_1?
(price_layer1/while/lstm_cell_4/Sigmoid_2Sigmoid-price_layer1/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2*
(price_layer1/while/lstm_cell_4/Sigmoid_2?
%price_layer1/while/lstm_cell_4/Relu_1Relu(price_layer1/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2'
%price_layer1/while/lstm_cell_4/Relu_1?
$price_layer1/while/lstm_cell_4/mul_2Mul,price_layer1/while/lstm_cell_4/Sigmoid_2:y:03price_layer1/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell_4/mul_2?
7price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer1_while_placeholder_1price_layer1_while_placeholder(price_layer1/while/lstm_cell_4/mul_2:z:0*
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
price_layer1/while/IdentityIdentityprice_layer1/while/add_1:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity?
price_layer1/while/Identity_1Identity8price_layer1_while_price_layer1_while_maximum_iterations6^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_1?
price_layer1/while/Identity_2Identityprice_layer1/while/add:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_2?
price_layer1/while/Identity_3IdentityGprice_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_3?
price_layer1/while/Identity_4Identity(price_layer1/while/lstm_cell_4/mul_2:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_4?
price_layer1/while/Identity_5Identity(price_layer1/while/lstm_cell_4/add_1:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_5"C
price_layer1_while_identity$price_layer1/while/Identity:output:0"G
price_layer1_while_identity_1&price_layer1/while/Identity_1:output:0"G
price_layer1_while_identity_2&price_layer1/while/Identity_2:output:0"G
price_layer1_while_identity_3&price_layer1/while/Identity_3:output:0"G
price_layer1_while_identity_4&price_layer1/while/Identity_4:output:0"G
price_layer1_while_identity_5&price_layer1/while/Identity_5:output:0"?
>price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource@price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0"?
?price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resourceAprice_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0"?
=price_layer1_while_lstm_cell_4_matmul_readvariableop_resource?price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0"d
/price_layer1_while_price_layer1_strided_slice_11price_layer1_while_price_layer1_strided_slice_1_0"?
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensormprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2n
5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp2l
4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp2p
6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
?B
?
while_body_595875563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??(while/lstm_cell_4/BiasAdd/ReadVariableOp?'while/lstm_cell_4/MatMul/ReadVariableOp?)while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
?Z
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875495

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??"lstm_cell_4/BiasAdd/ReadVariableOp?!lstm_cell_4/MatMul/ReadVariableOp?#lstm_cell_4/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
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
while_body_595875410* 
condR
while_cond_595875409*K
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
IdentityIdentitytranspose_1:y:0#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_model_4_layer_call_fn_595874262
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
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_5958742352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::22
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
??
?

F__inference_model_4_layer_call_and_return_conditional_losses_595874627
inputs_0
inputs_1;
7price_layer1_lstm_cell_4_matmul_readvariableop_resource=
9price_layer1_lstm_cell_4_matmul_1_readvariableop_resource<
8price_layer1_lstm_cell_4_biasadd_readvariableop_resource;
7price_layer2_lstm_cell_5_matmul_readvariableop_resource=
9price_layer2_lstm_cell_5_matmul_1_readvariableop_resource<
8price_layer2_lstm_cell_5_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp?.price_layer1/lstm_cell_4/MatMul/ReadVariableOp?0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp?price_layer1/while?/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp?.price_layer2/lstm_cell_5/MatMul/ReadVariableOp?0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp?price_layer2/while`
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
.price_layer1/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp7price_layer1_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype020
.price_layer1/lstm_cell_4/MatMul/ReadVariableOp?
price_layer1/lstm_cell_4/MatMulMatMul%price_layer1/strided_slice_2:output:06price_layer1/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_4/MatMul?
0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp9price_layer1_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype022
0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp?
!price_layer1/lstm_cell_4/MatMul_1MatMulprice_layer1/zeros:output:08price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2#
!price_layer1/lstm_cell_4/MatMul_1?
price_layer1/lstm_cell_4/addAddV2)price_layer1/lstm_cell_4/MatMul:product:0+price_layer1/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell_4/add?
/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp8price_layer1_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp?
 price_layer1/lstm_cell_4/BiasAddBiasAdd price_layer1/lstm_cell_4/add:z:07price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 price_layer1/lstm_cell_4/BiasAdd?
price_layer1/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
price_layer1/lstm_cell_4/Const?
(price_layer1/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(price_layer1/lstm_cell_4/split/split_dim?
price_layer1/lstm_cell_4/splitSplit1price_layer1/lstm_cell_4/split/split_dim:output:0)price_layer1/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2 
price_layer1/lstm_cell_4/split?
 price_layer1/lstm_cell_4/SigmoidSigmoid'price_layer1/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2"
 price_layer1/lstm_cell_4/Sigmoid?
"price_layer1/lstm_cell_4/Sigmoid_1Sigmoid'price_layer1/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2$
"price_layer1/lstm_cell_4/Sigmoid_1?
price_layer1/lstm_cell_4/mulMul&price_layer1/lstm_cell_4/Sigmoid_1:y:0price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell_4/mul?
price_layer1/lstm_cell_4/ReluRelu'price_layer1/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell_4/Relu?
price_layer1/lstm_cell_4/mul_1Mul$price_layer1/lstm_cell_4/Sigmoid:y:0+price_layer1/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell_4/mul_1?
price_layer1/lstm_cell_4/add_1AddV2 price_layer1/lstm_cell_4/mul:z:0"price_layer1/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell_4/add_1?
"price_layer1/lstm_cell_4/Sigmoid_2Sigmoid'price_layer1/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2$
"price_layer1/lstm_cell_4/Sigmoid_2?
price_layer1/lstm_cell_4/Relu_1Relu"price_layer1/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2!
price_layer1/lstm_cell_4/Relu_1?
price_layer1/lstm_cell_4/mul_2Mul&price_layer1/lstm_cell_4/Sigmoid_2:y:0-price_layer1/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell_4/mul_2?
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
price_layer1/whileWhile(price_layer1/while/loop_counter:output:0.price_layer1/while/maximum_iterations:output:0price_layer1/time:output:0%price_layer1/TensorArrayV2_1:handle:0price_layer1/zeros:output:0price_layer1/zeros_1:output:0%price_layer1/strided_slice_1:output:0Dprice_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:07price_layer1_lstm_cell_4_matmul_readvariableop_resource9price_layer1_lstm_cell_4_matmul_1_readvariableop_resource8price_layer1_lstm_cell_4_biasadd_readvariableop_resource*
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
!price_layer1_while_body_595874369*-
cond%R#
!price_layer1_while_cond_595874368*K
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
price_layer1/runtimet
price_layer2/ShapeShapeprice_layer1/transpose_1:y:0*
T0*
_output_shapes
:2
price_layer2/Shape?
 price_layer2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 price_layer2/strided_slice/stack?
"price_layer2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer2/strided_slice/stack_1?
"price_layer2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer2/strided_slice/stack_2?
price_layer2/strided_sliceStridedSliceprice_layer2/Shape:output:0)price_layer2/strided_slice/stack:output:0+price_layer2/strided_slice/stack_1:output:0+price_layer2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer2/strided_slicev
price_layer2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros/mul/y?
price_layer2/zeros/mulMul#price_layer2/strided_slice:output:0!price_layer2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros/muly
price_layer2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer2/zeros/Less/y?
price_layer2/zeros/LessLessprice_layer2/zeros/mul:z:0"price_layer2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros/Less|
price_layer2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros/packed/1?
price_layer2/zeros/packedPack#price_layer2/strided_slice:output:0$price_layer2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer2/zeros/packedy
price_layer2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer2/zeros/Const?
price_layer2/zerosFill"price_layer2/zeros/packed:output:0!price_layer2/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/zerosz
price_layer2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros_1/mul/y?
price_layer2/zeros_1/mulMul#price_layer2/strided_slice:output:0#price_layer2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros_1/mul}
price_layer2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer2/zeros_1/Less/y?
price_layer2/zeros_1/LessLessprice_layer2/zeros_1/mul:z:0$price_layer2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros_1/Less?
price_layer2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros_1/packed/1?
price_layer2/zeros_1/packedPack#price_layer2/strided_slice:output:0&price_layer2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer2/zeros_1/packed}
price_layer2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer2/zeros_1/Const?
price_layer2/zeros_1Fill$price_layer2/zeros_1/packed:output:0#price_layer2/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/zeros_1?
price_layer2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer2/transpose/perm?
price_layer2/transpose	Transposeprice_layer1/transpose_1:y:0$price_layer2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
price_layer2/transposev
price_layer2/Shape_1Shapeprice_layer2/transpose:y:0*
T0*
_output_shapes
:2
price_layer2/Shape_1?
"price_layer2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer2/strided_slice_1/stack?
$price_layer2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_1/stack_1?
$price_layer2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_1/stack_2?
price_layer2/strided_slice_1StridedSliceprice_layer2/Shape_1:output:0+price_layer2/strided_slice_1/stack:output:0-price_layer2/strided_slice_1/stack_1:output:0-price_layer2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer2/strided_slice_1?
(price_layer2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(price_layer2/TensorArrayV2/element_shape?
price_layer2/TensorArrayV2TensorListReserve1price_layer2/TensorArrayV2/element_shape:output:0%price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer2/TensorArrayV2?
Bprice_layer2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bprice_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape?
4price_layer2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorprice_layer2/transpose:y:0Kprice_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4price_layer2/TensorArrayUnstack/TensorListFromTensor?
"price_layer2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer2/strided_slice_2/stack?
$price_layer2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_2/stack_1?
$price_layer2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_2/stack_2?
price_layer2/strided_slice_2StridedSliceprice_layer2/transpose:y:0+price_layer2/strided_slice_2/stack:output:0-price_layer2/strided_slice_2/stack_1:output:0-price_layer2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
price_layer2/strided_slice_2?
.price_layer2/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp7price_layer2_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype020
.price_layer2/lstm_cell_5/MatMul/ReadVariableOp?
price_layer2/lstm_cell_5/MatMulMatMul%price_layer2/strided_slice_2:output:06price_layer2/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
price_layer2/lstm_cell_5/MatMul?
0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp9price_layer2_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype022
0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp?
!price_layer2/lstm_cell_5/MatMul_1MatMulprice_layer2/zeros:output:08price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!price_layer2/lstm_cell_5/MatMul_1?
price_layer2/lstm_cell_5/addAddV2)price_layer2/lstm_cell_5/MatMul:product:0+price_layer2/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
price_layer2/lstm_cell_5/add?
/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp8price_layer2_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp?
 price_layer2/lstm_cell_5/BiasAddBiasAdd price_layer2/lstm_cell_5/add:z:07price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 price_layer2/lstm_cell_5/BiasAdd?
price_layer2/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
price_layer2/lstm_cell_5/Const?
(price_layer2/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(price_layer2/lstm_cell_5/split/split_dim?
price_layer2/lstm_cell_5/splitSplit1price_layer2/lstm_cell_5/split/split_dim:output:0)price_layer2/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
price_layer2/lstm_cell_5/split?
 price_layer2/lstm_cell_5/SigmoidSigmoid'price_layer2/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2"
 price_layer2/lstm_cell_5/Sigmoid?
"price_layer2/lstm_cell_5/Sigmoid_1Sigmoid'price_layer2/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2$
"price_layer2/lstm_cell_5/Sigmoid_1?
price_layer2/lstm_cell_5/mulMul&price_layer2/lstm_cell_5/Sigmoid_1:y:0price_layer2/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/lstm_cell_5/mul?
price_layer2/lstm_cell_5/ReluRelu'price_layer2/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
price_layer2/lstm_cell_5/Relu?
price_layer2/lstm_cell_5/mul_1Mul$price_layer2/lstm_cell_5/Sigmoid:y:0+price_layer2/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_5/mul_1?
price_layer2/lstm_cell_5/add_1AddV2 price_layer2/lstm_cell_5/mul:z:0"price_layer2/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_5/add_1?
"price_layer2/lstm_cell_5/Sigmoid_2Sigmoid'price_layer2/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2$
"price_layer2/lstm_cell_5/Sigmoid_2?
price_layer2/lstm_cell_5/Relu_1Relu"price_layer2/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_5/Relu_1?
price_layer2/lstm_cell_5/mul_2Mul&price_layer2/lstm_cell_5/Sigmoid_2:y:0-price_layer2/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_5/mul_2?
*price_layer2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2,
*price_layer2/TensorArrayV2_1/element_shape?
price_layer2/TensorArrayV2_1TensorListReserve3price_layer2/TensorArrayV2_1/element_shape:output:0%price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer2/TensorArrayV2_1h
price_layer2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/time?
%price_layer2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%price_layer2/while/maximum_iterations?
price_layer2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
price_layer2/while/loop_counter?
price_layer2/whileWhile(price_layer2/while/loop_counter:output:0.price_layer2/while/maximum_iterations:output:0price_layer2/time:output:0%price_layer2/TensorArrayV2_1:handle:0price_layer2/zeros:output:0price_layer2/zeros_1:output:0%price_layer2/strided_slice_1:output:0Dprice_layer2/TensorArrayUnstack/TensorListFromTensor:output_handle:07price_layer2_lstm_cell_5_matmul_readvariableop_resource9price_layer2_lstm_cell_5_matmul_1_readvariableop_resource8price_layer2_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!price_layer2_while_body_595874518*-
cond%R#
!price_layer2_while_cond_595874517*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
price_layer2/while?
=price_layer2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2?
=price_layer2/TensorArrayV2Stack/TensorListStack/element_shape?
/price_layer2/TensorArrayV2Stack/TensorListStackTensorListStackprice_layer2/while:output:3Fprice_layer2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype021
/price_layer2/TensorArrayV2Stack/TensorListStack?
"price_layer2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"price_layer2/strided_slice_3/stack?
$price_layer2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$price_layer2/strided_slice_3/stack_1?
$price_layer2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_3/stack_2?
price_layer2/strided_slice_3StridedSlice8price_layer2/TensorArrayV2Stack/TensorListStack:tensor:0+price_layer2/strided_slice_3/stack:output:0-price_layer2/strided_slice_3/stack_1:output:0-price_layer2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
price_layer2/strided_slice_3?
price_layer2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer2/transpose_1/perm?
price_layer2/transpose_1	Transpose8price_layer2/TensorArrayV2Stack/TensorListStack:tensor:0&price_layer2/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
price_layer2/transpose_1?
price_layer2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer2/runtime{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
price_flatten/Const?
price_flatten/ReshapeReshape%price_layer2/strided_slice_3:output:0price_flatten/Const:output:0*
T0*'
_output_shapes
:????????? 2
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
:?????????"2
concat_layer/concat?
"fixed_layer1/MatMul/ReadVariableOpReadVariableOp+fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:"*
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
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp0^price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp/^price_layer1/lstm_cell_4/MatMul/ReadVariableOp1^price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp^price_layer1/while0^price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp/^price_layer2/lstm_cell_5/MatMul/ReadVariableOp1^price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp^price_layer2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2b
/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp2`
.price_layer1/lstm_cell_4/MatMul/ReadVariableOp.price_layer1/lstm_cell_4/MatMul/ReadVariableOp2d
0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp2(
price_layer1/whileprice_layer1/while2b
/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp2`
.price_layer2/lstm_cell_5/MatMul/ReadVariableOp.price_layer2/lstm_cell_5/MatMul/ReadVariableOp2d
0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp2(
price_layer2/whileprice_layer2/while:U Q
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
?B
?
while_body_595875738
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?B
?
while_body_595873867
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?B
?
while_body_595876066
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
??
?

F__inference_model_4_layer_call_and_return_conditional_losses_595874954
inputs_0
inputs_1;
7price_layer1_lstm_cell_4_matmul_readvariableop_resource=
9price_layer1_lstm_cell_4_matmul_1_readvariableop_resource<
8price_layer1_lstm_cell_4_biasadd_readvariableop_resource;
7price_layer2_lstm_cell_5_matmul_readvariableop_resource=
9price_layer2_lstm_cell_5_matmul_1_readvariableop_resource<
8price_layer2_lstm_cell_5_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp?.price_layer1/lstm_cell_4/MatMul/ReadVariableOp?0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp?price_layer1/while?/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp?.price_layer2/lstm_cell_5/MatMul/ReadVariableOp?0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp?price_layer2/while`
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
.price_layer1/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp7price_layer1_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype020
.price_layer1/lstm_cell_4/MatMul/ReadVariableOp?
price_layer1/lstm_cell_4/MatMulMatMul%price_layer1/strided_slice_2:output:06price_layer1/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_4/MatMul?
0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp9price_layer1_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype022
0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp?
!price_layer1/lstm_cell_4/MatMul_1MatMulprice_layer1/zeros:output:08price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2#
!price_layer1/lstm_cell_4/MatMul_1?
price_layer1/lstm_cell_4/addAddV2)price_layer1/lstm_cell_4/MatMul:product:0+price_layer1/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell_4/add?
/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp8price_layer1_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp?
 price_layer1/lstm_cell_4/BiasAddBiasAdd price_layer1/lstm_cell_4/add:z:07price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2"
 price_layer1/lstm_cell_4/BiasAdd?
price_layer1/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
price_layer1/lstm_cell_4/Const?
(price_layer1/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(price_layer1/lstm_cell_4/split/split_dim?
price_layer1/lstm_cell_4/splitSplit1price_layer1/lstm_cell_4/split/split_dim:output:0)price_layer1/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2 
price_layer1/lstm_cell_4/split?
 price_layer1/lstm_cell_4/SigmoidSigmoid'price_layer1/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2"
 price_layer1/lstm_cell_4/Sigmoid?
"price_layer1/lstm_cell_4/Sigmoid_1Sigmoid'price_layer1/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2$
"price_layer1/lstm_cell_4/Sigmoid_1?
price_layer1/lstm_cell_4/mulMul&price_layer1/lstm_cell_4/Sigmoid_1:y:0price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell_4/mul?
price_layer1/lstm_cell_4/ReluRelu'price_layer1/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
price_layer1/lstm_cell_4/Relu?
price_layer1/lstm_cell_4/mul_1Mul$price_layer1/lstm_cell_4/Sigmoid:y:0+price_layer1/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell_4/mul_1?
price_layer1/lstm_cell_4/add_1AddV2 price_layer1/lstm_cell_4/mul:z:0"price_layer1/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell_4/add_1?
"price_layer1/lstm_cell_4/Sigmoid_2Sigmoid'price_layer1/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2$
"price_layer1/lstm_cell_4/Sigmoid_2?
price_layer1/lstm_cell_4/Relu_1Relu"price_layer1/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2!
price_layer1/lstm_cell_4/Relu_1?
price_layer1/lstm_cell_4/mul_2Mul&price_layer1/lstm_cell_4/Sigmoid_2:y:0-price_layer1/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2 
price_layer1/lstm_cell_4/mul_2?
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
price_layer1/whileWhile(price_layer1/while/loop_counter:output:0.price_layer1/while/maximum_iterations:output:0price_layer1/time:output:0%price_layer1/TensorArrayV2_1:handle:0price_layer1/zeros:output:0price_layer1/zeros_1:output:0%price_layer1/strided_slice_1:output:0Dprice_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:07price_layer1_lstm_cell_4_matmul_readvariableop_resource9price_layer1_lstm_cell_4_matmul_1_readvariableop_resource8price_layer1_lstm_cell_4_biasadd_readvariableop_resource*
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
!price_layer1_while_body_595874696*-
cond%R#
!price_layer1_while_cond_595874695*K
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
price_layer1/runtimet
price_layer2/ShapeShapeprice_layer1/transpose_1:y:0*
T0*
_output_shapes
:2
price_layer2/Shape?
 price_layer2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 price_layer2/strided_slice/stack?
"price_layer2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer2/strided_slice/stack_1?
"price_layer2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"price_layer2/strided_slice/stack_2?
price_layer2/strided_sliceStridedSliceprice_layer2/Shape:output:0)price_layer2/strided_slice/stack:output:0+price_layer2/strided_slice/stack_1:output:0+price_layer2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer2/strided_slicev
price_layer2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros/mul/y?
price_layer2/zeros/mulMul#price_layer2/strided_slice:output:0!price_layer2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros/muly
price_layer2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer2/zeros/Less/y?
price_layer2/zeros/LessLessprice_layer2/zeros/mul:z:0"price_layer2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros/Less|
price_layer2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros/packed/1?
price_layer2/zeros/packedPack#price_layer2/strided_slice:output:0$price_layer2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer2/zeros/packedy
price_layer2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer2/zeros/Const?
price_layer2/zerosFill"price_layer2/zeros/packed:output:0!price_layer2/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/zerosz
price_layer2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros_1/mul/y?
price_layer2/zeros_1/mulMul#price_layer2/strided_slice:output:0#price_layer2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros_1/mul}
price_layer2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
price_layer2/zeros_1/Less/y?
price_layer2/zeros_1/LessLessprice_layer2/zeros_1/mul:z:0$price_layer2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
price_layer2/zeros_1/Less?
price_layer2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/zeros_1/packed/1?
price_layer2/zeros_1/packedPack#price_layer2/strided_slice:output:0&price_layer2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
price_layer2/zeros_1/packed}
price_layer2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer2/zeros_1/Const?
price_layer2/zeros_1Fill$price_layer2/zeros_1/packed:output:0#price_layer2/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/zeros_1?
price_layer2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer2/transpose/perm?
price_layer2/transpose	Transposeprice_layer1/transpose_1:y:0$price_layer2/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2
price_layer2/transposev
price_layer2/Shape_1Shapeprice_layer2/transpose:y:0*
T0*
_output_shapes
:2
price_layer2/Shape_1?
"price_layer2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer2/strided_slice_1/stack?
$price_layer2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_1/stack_1?
$price_layer2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_1/stack_2?
price_layer2/strided_slice_1StridedSliceprice_layer2/Shape_1:output:0+price_layer2/strided_slice_1/stack:output:0-price_layer2/strided_slice_1/stack_1:output:0-price_layer2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
price_layer2/strided_slice_1?
(price_layer2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(price_layer2/TensorArrayV2/element_shape?
price_layer2/TensorArrayV2TensorListReserve1price_layer2/TensorArrayV2/element_shape:output:0%price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer2/TensorArrayV2?
Bprice_layer2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2D
Bprice_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape?
4price_layer2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorprice_layer2/transpose:y:0Kprice_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type026
4price_layer2/TensorArrayUnstack/TensorListFromTensor?
"price_layer2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"price_layer2/strided_slice_2/stack?
$price_layer2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_2/stack_1?
$price_layer2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_2/stack_2?
price_layer2/strided_slice_2StridedSliceprice_layer2/transpose:y:0+price_layer2/strided_slice_2/stack:output:0-price_layer2/strided_slice_2/stack_1:output:0-price_layer2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2
price_layer2/strided_slice_2?
.price_layer2/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp7price_layer2_lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype020
.price_layer2/lstm_cell_5/MatMul/ReadVariableOp?
price_layer2/lstm_cell_5/MatMulMatMul%price_layer2/strided_slice_2:output:06price_layer2/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2!
price_layer2/lstm_cell_5/MatMul?
0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp9price_layer2_lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype022
0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp?
!price_layer2/lstm_cell_5/MatMul_1MatMulprice_layer2/zeros:output:08price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!price_layer2/lstm_cell_5/MatMul_1?
price_layer2/lstm_cell_5/addAddV2)price_layer2/lstm_cell_5/MatMul:product:0+price_layer2/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
price_layer2/lstm_cell_5/add?
/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp8price_layer2_lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp?
 price_layer2/lstm_cell_5/BiasAddBiasAdd price_layer2/lstm_cell_5/add:z:07price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 price_layer2/lstm_cell_5/BiasAdd?
price_layer2/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2 
price_layer2/lstm_cell_5/Const?
(price_layer2/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(price_layer2/lstm_cell_5/split/split_dim?
price_layer2/lstm_cell_5/splitSplit1price_layer2/lstm_cell_5/split/split_dim:output:0)price_layer2/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2 
price_layer2/lstm_cell_5/split?
 price_layer2/lstm_cell_5/SigmoidSigmoid'price_layer2/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2"
 price_layer2/lstm_cell_5/Sigmoid?
"price_layer2/lstm_cell_5/Sigmoid_1Sigmoid'price_layer2/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2$
"price_layer2/lstm_cell_5/Sigmoid_1?
price_layer2/lstm_cell_5/mulMul&price_layer2/lstm_cell_5/Sigmoid_1:y:0price_layer2/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/lstm_cell_5/mul?
price_layer2/lstm_cell_5/ReluRelu'price_layer2/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
price_layer2/lstm_cell_5/Relu?
price_layer2/lstm_cell_5/mul_1Mul$price_layer2/lstm_cell_5/Sigmoid:y:0+price_layer2/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_5/mul_1?
price_layer2/lstm_cell_5/add_1AddV2 price_layer2/lstm_cell_5/mul:z:0"price_layer2/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_5/add_1?
"price_layer2/lstm_cell_5/Sigmoid_2Sigmoid'price_layer2/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2$
"price_layer2/lstm_cell_5/Sigmoid_2?
price_layer2/lstm_cell_5/Relu_1Relu"price_layer2/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_5/Relu_1?
price_layer2/lstm_cell_5/mul_2Mul&price_layer2/lstm_cell_5/Sigmoid_2:y:0-price_layer2/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_5/mul_2?
*price_layer2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2,
*price_layer2/TensorArrayV2_1/element_shape?
price_layer2/TensorArrayV2_1TensorListReserve3price_layer2/TensorArrayV2_1/element_shape:output:0%price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
price_layer2/TensorArrayV2_1h
price_layer2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
price_layer2/time?
%price_layer2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%price_layer2/while/maximum_iterations?
price_layer2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2!
price_layer2/while/loop_counter?
price_layer2/whileWhile(price_layer2/while/loop_counter:output:0.price_layer2/while/maximum_iterations:output:0price_layer2/time:output:0%price_layer2/TensorArrayV2_1:handle:0price_layer2/zeros:output:0price_layer2/zeros_1:output:0%price_layer2/strided_slice_1:output:0Dprice_layer2/TensorArrayUnstack/TensorListFromTensor:output_handle:07price_layer2_lstm_cell_5_matmul_readvariableop_resource9price_layer2_lstm_cell_5_matmul_1_readvariableop_resource8price_layer2_lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*-
body%R#
!price_layer2_while_body_595874845*-
cond%R#
!price_layer2_while_cond_595874844*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
price_layer2/while?
=price_layer2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2?
=price_layer2/TensorArrayV2Stack/TensorListStack/element_shape?
/price_layer2/TensorArrayV2Stack/TensorListStackTensorListStackprice_layer2/while:output:3Fprice_layer2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype021
/price_layer2/TensorArrayV2Stack/TensorListStack?
"price_layer2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2$
"price_layer2/strided_slice_3/stack?
$price_layer2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$price_layer2/strided_slice_3/stack_1?
$price_layer2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$price_layer2/strided_slice_3/stack_2?
price_layer2/strided_slice_3StridedSlice8price_layer2/TensorArrayV2Stack/TensorListStack:tensor:0+price_layer2/strided_slice_3/stack:output:0-price_layer2/strided_slice_3/stack_1:output:0-price_layer2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2
price_layer2/strided_slice_3?
price_layer2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
price_layer2/transpose_1/perm?
price_layer2/transpose_1	Transpose8price_layer2/TensorArrayV2Stack/TensorListStack:tensor:0&price_layer2/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2
price_layer2/transpose_1?
price_layer2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
price_layer2/runtime{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
price_flatten/Const?
price_flatten/ReshapeReshape%price_layer2/strided_slice_3:output:0price_flatten/Const:output:0*
T0*'
_output_shapes
:????????? 2
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
:?????????"2
concat_layer/concat?
"fixed_layer1/MatMul/ReadVariableOpReadVariableOp+fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:"*
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
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp0^price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp/^price_layer1/lstm_cell_4/MatMul/ReadVariableOp1^price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp^price_layer1/while0^price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp/^price_layer2/lstm_cell_5/MatMul/ReadVariableOp1^price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp^price_layer2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2b
/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp/price_layer1/lstm_cell_4/BiasAdd/ReadVariableOp2`
.price_layer1/lstm_cell_4/MatMul/ReadVariableOp.price_layer1/lstm_cell_4/MatMul/ReadVariableOp2d
0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp0price_layer1/lstm_cell_4/MatMul_1/ReadVariableOp2(
price_layer1/whileprice_layer1/while2b
/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp/price_layer2/lstm_cell_5/BiasAdd/ReadVariableOp2`
.price_layer2/lstm_cell_5/MatMul/ReadVariableOp.price_layer2/lstm_cell_5/MatMul/ReadVariableOp2d
0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp0price_layer2/lstm_cell_5/MatMul_1/ReadVariableOp2(
price_layer2/whileprice_layer2/while:U Q
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
?
?
while_cond_595873531
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595873531___redundant_placeholder07
3while_while_cond_595873531___redundant_placeholder17
3while_while_cond_595873531___redundant_placeholder27
3while_while_cond_595873531___redundant_placeholder3
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

?
+__inference_model_4_layer_call_fn_595875014
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
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_5958742352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::22
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
?
!price_layer2_while_cond_5958748446
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_38
4price_layer2_while_less_price_layer2_strided_slice_1Q
Mprice_layer2_while_price_layer2_while_cond_595874844___redundant_placeholder0Q
Mprice_layer2_while_price_layer2_while_cond_595874844___redundant_placeholder1Q
Mprice_layer2_while_price_layer2_while_cond_595874844___redundant_placeholder2Q
Mprice_layer2_while_price_layer2_while_cond_595874844___redundant_placeholder3
price_layer2_while_identity
?
price_layer2/while/LessLessprice_layer2_while_placeholder4price_layer2_while_less_price_layer2_strided_slice_1*
T0*
_output_shapes
: 2
price_layer2/while/Less?
price_layer2/while/IdentityIdentityprice_layer2/while/Less:z:0*
T0
*
_output_shapes
: 2
price_layer2/while/Identity"C
price_layer2_while_identity$price_layer2/while/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?D
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595872688

inputs
lstm_cell_4_595872606
lstm_cell_4_595872608
lstm_cell_4_595872610
identity??#lstm_cell_4/StatefulPartitionedCall?whileD
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
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_595872606lstm_cell_4_595872608lstm_cell_4_595872610*
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
GPU 2J 8? *S
fNRL
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_5958721932%
#lstm_cell_4/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_595872606lstm_cell_4_595872608lstm_cell_4_595872610*
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
while_body_595872619* 
condR
while_cond_595872618*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_4/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
)model_4_price_layer2_while_cond_595871977F
Bmodel_4_price_layer2_while_model_4_price_layer2_while_loop_counterL
Hmodel_4_price_layer2_while_model_4_price_layer2_while_maximum_iterations*
&model_4_price_layer2_while_placeholder,
(model_4_price_layer2_while_placeholder_1,
(model_4_price_layer2_while_placeholder_2,
(model_4_price_layer2_while_placeholder_3H
Dmodel_4_price_layer2_while_less_model_4_price_layer2_strided_slice_1a
]model_4_price_layer2_while_model_4_price_layer2_while_cond_595871977___redundant_placeholder0a
]model_4_price_layer2_while_model_4_price_layer2_while_cond_595871977___redundant_placeholder1a
]model_4_price_layer2_while_model_4_price_layer2_while_cond_595871977___redundant_placeholder2a
]model_4_price_layer2_while_model_4_price_layer2_while_cond_595871977___redundant_placeholder3'
#model_4_price_layer2_while_identity
?
model_4/price_layer2/while/LessLess&model_4_price_layer2_while_placeholderDmodel_4_price_layer2_while_less_model_4_price_layer2_strided_slice_1*
T0*
_output_shapes
: 2!
model_4/price_layer2/while/Less?
#model_4/price_layer2/while/IdentityIdentity#model_4/price_layer2/while/Less:z:0*
T0
*
_output_shapes
: 2%
#model_4/price_layer2/while/Identity"S
#model_4_price_layer2_while_identity,model_4/price_layer2/while/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_595876332

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_595874023

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:"*
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
:?????????"::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????"
 
_user_specified_nameinputs
?M
?

%__inference__traced_restore_595876751
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
assignvariableop_9_sgd_momentum7
3assignvariableop_10_price_layer1_lstm_cell_4_kernelA
=assignvariableop_11_price_layer1_lstm_cell_4_recurrent_kernel5
1assignvariableop_12_price_layer1_lstm_cell_4_bias7
3assignvariableop_13_price_layer2_lstm_cell_5_kernelA
=assignvariableop_14_price_layer2_lstm_cell_5_recurrent_kernel5
1assignvariableop_15_price_layer2_lstm_cell_5_bias
assignvariableop_16_total
assignvariableop_17_count
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
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
AssignVariableOp_10AssignVariableOp3assignvariableop_10_price_layer1_lstm_cell_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp=assignvariableop_11_price_layer1_lstm_cell_4_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp1assignvariableop_12_price_layer1_lstm_cell_4_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp3assignvariableop_13_price_layer2_lstm_cell_5_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp=assignvariableop_14_price_layer2_lstm_cell_5_recurrent_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_price_layer2_lstm_cell_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18?
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_595874076

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
?&
?
F__inference_model_4_layer_call_and_return_conditional_losses_595874093
price_input
	env_input
price_layer1_595873640
price_layer1_595873642
price_layer1_595873644
price_layer2_595873975
price_layer2_595873977
price_layer2_595873979
fixed_layer1_595874034
fixed_layer1_595874036
fixed_layer2_595874061
fixed_layer2_595874063
action_output_595874087
action_output_595874089
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_595873640price_layer1_595873642price_layer1_595873644*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958734642&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_595873975price_layer2_595873977price_layer2_595873979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958737992&
$price_layer2/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_5958739882
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0	env_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_5958740032
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_595874034fixed_layer1_595874036*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_5958740232&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_595874061fixed_layer2_595874063*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_5958740502&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_595874087action_output_595874089*
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
L__inference_action_output_layer_call_and_return_conditional_losses_5958740762'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2N
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
?%
?
while_body_595872487
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_4_595872511_0!
while_lstm_cell_4_595872513_0!
while_lstm_cell_4_595872515_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4_595872511
while_lstm_cell_4_595872513
while_lstm_cell_4_595872515??)while/lstm_cell_4/StatefulPartitionedCall?
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
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_595872511_0while_lstm_cell_4_595872513_0while_lstm_cell_4_595872515_0*
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
GPU 2J 8? *S
fNRL
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_5958721602+
)while/lstm_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_4_595872511while_lstm_cell_4_595872511_0"<
while_lstm_cell_4_595872513while_lstm_cell_4_595872513_0"<
while_lstm_cell_4_595872515while_lstm_cell_4_595872515_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 
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
?
!price_layer1_while_cond_5958746956
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_38
4price_layer1_while_less_price_layer1_strided_slice_1Q
Mprice_layer1_while_price_layer1_while_cond_595874695___redundant_placeholder0Q
Mprice_layer1_while_price_layer1_while_cond_595874695___redundant_placeholder1Q
Mprice_layer1_while_price_layer1_while_cond_595874695___redundant_placeholder2Q
Mprice_layer1_while_price_layer1_while_cond_595874695___redundant_placeholder3
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
?Z
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595873464

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??"lstm_cell_4/BiasAdd/ReadVariableOp?!lstm_cell_4/MatMul/ReadVariableOp?#lstm_cell_4/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
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
while_body_595873379* 
condR
while_cond_595873378*K
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
IdentityIdentitytranspose_1:y:0#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?U
?
!price_layer1_while_body_5958743696
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_35
1price_layer1_while_price_layer1_strided_slice_1_0q
mprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0C
?price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0E
Aprice_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0D
@price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0
price_layer1_while_identity!
price_layer1_while_identity_1!
price_layer1_while_identity_2!
price_layer1_while_identity_3!
price_layer1_while_identity_4!
price_layer1_while_identity_53
/price_layer1_while_price_layer1_strided_slice_1o
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensorA
=price_layer1_while_lstm_cell_4_matmul_readvariableop_resourceC
?price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resourceB
>price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource??5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp?4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp?price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype026
4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp?
%price_layer1/while/lstm_cell_4/MatMulMatMul=price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0<price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_4/MatMul?
6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpAprice_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype028
6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp?
'price_layer1/while/lstm_cell_4/MatMul_1MatMul price_layer1_while_placeholder_2>price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'price_layer1/while/lstm_cell_4/MatMul_1?
"price_layer1/while/lstm_cell_4/addAddV2/price_layer1/while/lstm_cell_4/MatMul:product:01price_layer1/while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2$
"price_layer1/while/lstm_cell_4/add?
5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp@price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype027
5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp?
&price_layer1/while/lstm_cell_4/BiasAddBiasAdd&price_layer1/while/lstm_cell_4/add:z:0=price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2(
&price_layer1/while/lstm_cell_4/BiasAdd?
$price_layer1/while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$price_layer1/while/lstm_cell_4/Const?
.price_layer1/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.price_layer1/while/lstm_cell_4/split/split_dim?
$price_layer1/while/lstm_cell_4/splitSplit7price_layer1/while/lstm_cell_4/split/split_dim:output:0/price_layer1/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2&
$price_layer1/while/lstm_cell_4/split?
&price_layer1/while/lstm_cell_4/SigmoidSigmoid-price_layer1/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2(
&price_layer1/while/lstm_cell_4/Sigmoid?
(price_layer1/while/lstm_cell_4/Sigmoid_1Sigmoid-price_layer1/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2*
(price_layer1/while/lstm_cell_4/Sigmoid_1?
"price_layer1/while/lstm_cell_4/mulMul,price_layer1/while/lstm_cell_4/Sigmoid_1:y:0 price_layer1_while_placeholder_3*
T0*'
_output_shapes
:?????????2$
"price_layer1/while/lstm_cell_4/mul?
#price_layer1/while/lstm_cell_4/ReluRelu-price_layer1/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2%
#price_layer1/while/lstm_cell_4/Relu?
$price_layer1/while/lstm_cell_4/mul_1Mul*price_layer1/while/lstm_cell_4/Sigmoid:y:01price_layer1/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell_4/mul_1?
$price_layer1/while/lstm_cell_4/add_1AddV2&price_layer1/while/lstm_cell_4/mul:z:0(price_layer1/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell_4/add_1?
(price_layer1/while/lstm_cell_4/Sigmoid_2Sigmoid-price_layer1/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2*
(price_layer1/while/lstm_cell_4/Sigmoid_2?
%price_layer1/while/lstm_cell_4/Relu_1Relu(price_layer1/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2'
%price_layer1/while/lstm_cell_4/Relu_1?
$price_layer1/while/lstm_cell_4/mul_2Mul,price_layer1/while/lstm_cell_4/Sigmoid_2:y:03price_layer1/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2&
$price_layer1/while/lstm_cell_4/mul_2?
7price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer1_while_placeholder_1price_layer1_while_placeholder(price_layer1/while/lstm_cell_4/mul_2:z:0*
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
price_layer1/while/IdentityIdentityprice_layer1/while/add_1:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity?
price_layer1/while/Identity_1Identity8price_layer1_while_price_layer1_while_maximum_iterations6^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_1?
price_layer1/while/Identity_2Identityprice_layer1/while/add:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_2?
price_layer1/while/Identity_3IdentityGprice_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_3?
price_layer1/while/Identity_4Identity(price_layer1/while/lstm_cell_4/mul_2:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_4?
price_layer1/while/Identity_5Identity(price_layer1/while/lstm_cell_4/add_1:z:06^price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5^price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp7^price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
price_layer1/while/Identity_5"C
price_layer1_while_identity$price_layer1/while/Identity:output:0"G
price_layer1_while_identity_1&price_layer1/while/Identity_1:output:0"G
price_layer1_while_identity_2&price_layer1/while/Identity_2:output:0"G
price_layer1_while_identity_3&price_layer1/while/Identity_3:output:0"G
price_layer1_while_identity_4&price_layer1/while/Identity_4:output:0"G
price_layer1_while_identity_5&price_layer1/while/Identity_5:output:0"?
>price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource@price_layer1_while_lstm_cell_4_biasadd_readvariableop_resource_0"?
?price_layer1_while_lstm_cell_4_matmul_1_readvariableop_resourceAprice_layer1_while_lstm_cell_4_matmul_1_readvariableop_resource_0"?
=price_layer1_while_lstm_cell_4_matmul_readvariableop_resource?price_layer1_while_lstm_cell_4_matmul_readvariableop_resource_0"d
/price_layer1_while_price_layer1_strided_slice_11price_layer1_while_price_layer1_strided_slice_1_0"?
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensormprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2n
5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp5price_layer1/while/lstm_cell_4/BiasAdd/ReadVariableOp2l
4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp4price_layer1/while/lstm_cell_4/MatMul/ReadVariableOp2p
6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp6price_layer1/while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
?
w
K__inference_concat_layer_layer_call_and_return_conditional_losses_595876344
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
:?????????"2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?B
?
while_body_595875082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??(while/lstm_cell_4/BiasAdd/ReadVariableOp?'while/lstm_cell_4/MatMul/ReadVariableOp?)while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_595876381

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
?B
?
while_body_595875891
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_5_matmul_readvariableop_resource_08
4while_lstm_cell_5_matmul_1_readvariableop_resource_07
3while_lstm_cell_5_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_5_matmul_readvariableop_resource6
2while_lstm_cell_5_matmul_1_readvariableop_resource5
1while_lstm_cell_5_biasadd_readvariableop_resource??(while/lstm_cell_5/BiasAdd/ReadVariableOp?'while/lstm_cell_5/MatMul/ReadVariableOp?)while/lstm_cell_5/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
'while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02)
'while/lstm_cell_5/MatMul/ReadVariableOp?
while/lstm_cell_5/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul?
)while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02+
)while/lstm_cell_5/MatMul_1/ReadVariableOp?
while/lstm_cell_5/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/MatMul_1?
while/lstm_cell_5/addAddV2"while/lstm_cell_5/MatMul:product:0$while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/add?
(while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02*
(while/lstm_cell_5/BiasAdd/ReadVariableOp?
while/lstm_cell_5/BiasAddBiasAddwhile/lstm_cell_5/add:z:00while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_5/BiasAddt
while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_5/Const?
!while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_5/split/split_dim?
while/lstm_cell_5/splitSplit*while/lstm_cell_5/split/split_dim:output:0"while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_5/split?
while/lstm_cell_5/SigmoidSigmoid while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid?
while/lstm_cell_5/Sigmoid_1Sigmoid while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_1?
while/lstm_cell_5/mulMulwhile/lstm_cell_5/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul?
while/lstm_cell_5/ReluRelu while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu?
while/lstm_cell_5/mul_1Mulwhile/lstm_cell_5/Sigmoid:y:0$while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_1?
while/lstm_cell_5/add_1AddV2while/lstm_cell_5/mul:z:0while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/add_1?
while/lstm_cell_5/Sigmoid_2Sigmoid while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Sigmoid_2?
while/lstm_cell_5/Relu_1Reluwhile/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/Relu_1?
while/lstm_cell_5/mul_2Mulwhile/lstm_cell_5/Sigmoid_2:y:0&while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_5/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_5/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_5/mul_2:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_5/add_1:z:0)^while/lstm_cell_5/BiasAdd/ReadVariableOp(^while/lstm_cell_5/MatMul/ReadVariableOp*^while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_5_biasadd_readvariableop_resource3while_lstm_cell_5_biasadd_readvariableop_resource_0"j
2while_lstm_cell_5_matmul_1_readvariableop_resource4while_lstm_cell_5_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_5_matmul_readvariableop_resource2while_lstm_cell_5_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2T
(while/lstm_cell_5/BiasAdd/ReadVariableOp(while/lstm_cell_5/BiasAdd/ReadVariableOp2R
'while/lstm_cell_5/MatMul/ReadVariableOp'while/lstm_cell_5/MatMul/ReadVariableOp2V
)while/lstm_cell_5/MatMul_1/ReadVariableOp)while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_595873988

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:????????? 2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
0__inference_price_layer2_layer_call_fn_595875998
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
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958732982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?U
?
!price_layer2_while_body_5958748456
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_35
1price_layer2_while_price_layer2_strided_slice_1_0q
mprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0C
?price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0E
Aprice_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0D
@price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0
price_layer2_while_identity!
price_layer2_while_identity_1!
price_layer2_while_identity_2!
price_layer2_while_identity_3!
price_layer2_while_identity_4!
price_layer2_while_identity_53
/price_layer2_while_price_layer2_strided_slice_1o
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensorA
=price_layer2_while_lstm_cell_5_matmul_readvariableop_resourceC
?price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resourceB
>price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource??5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp?4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp?
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6price_layer2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0price_layer2_while_placeholderMprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6price_layer2/while/TensorArrayV2Read/TensorListGetItem?
4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp?price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype026
4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?
%price_layer2/while/lstm_cell_5/MatMulMatMul=price_layer2/while/TensorArrayV2Read/TensorListGetItem:item:0<price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%price_layer2/while/lstm_cell_5/MatMul?
6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpAprice_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype028
6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp?
'price_layer2/while/lstm_cell_5/MatMul_1MatMul price_layer2_while_placeholder_2>price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'price_layer2/while/lstm_cell_5/MatMul_1?
"price_layer2/while/lstm_cell_5/addAddV2/price_layer2/while/lstm_cell_5/MatMul:product:01price_layer2/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"price_layer2/while/lstm_cell_5/add?
5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp@price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype027
5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp?
&price_layer2/while/lstm_cell_5/BiasAddBiasAdd&price_layer2/while/lstm_cell_5/add:z:0=price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&price_layer2/while/lstm_cell_5/BiasAdd?
$price_layer2/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$price_layer2/while/lstm_cell_5/Const?
.price_layer2/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.price_layer2/while/lstm_cell_5/split/split_dim?
$price_layer2/while/lstm_cell_5/splitSplit7price_layer2/while/lstm_cell_5/split/split_dim:output:0/price_layer2/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2&
$price_layer2/while/lstm_cell_5/split?
&price_layer2/while/lstm_cell_5/SigmoidSigmoid-price_layer2/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2(
&price_layer2/while/lstm_cell_5/Sigmoid?
(price_layer2/while/lstm_cell_5/Sigmoid_1Sigmoid-price_layer2/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2*
(price_layer2/while/lstm_cell_5/Sigmoid_1?
"price_layer2/while/lstm_cell_5/mulMul,price_layer2/while/lstm_cell_5/Sigmoid_1:y:0 price_layer2_while_placeholder_3*
T0*'
_output_shapes
:????????? 2$
"price_layer2/while/lstm_cell_5/mul?
#price_layer2/while/lstm_cell_5/ReluRelu-price_layer2/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2%
#price_layer2/while/lstm_cell_5/Relu?
$price_layer2/while/lstm_cell_5/mul_1Mul*price_layer2/while/lstm_cell_5/Sigmoid:y:01price_layer2/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_5/mul_1?
$price_layer2/while/lstm_cell_5/add_1AddV2&price_layer2/while/lstm_cell_5/mul:z:0(price_layer2/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_5/add_1?
(price_layer2/while/lstm_cell_5/Sigmoid_2Sigmoid-price_layer2/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2*
(price_layer2/while/lstm_cell_5/Sigmoid_2?
%price_layer2/while/lstm_cell_5/Relu_1Relu(price_layer2/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_5/Relu_1?
$price_layer2/while/lstm_cell_5/mul_2Mul,price_layer2/while/lstm_cell_5/Sigmoid_2:y:03price_layer2/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_5/mul_2?
7price_layer2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer2_while_placeholder_1price_layer2_while_placeholder(price_layer2/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype029
7price_layer2/while/TensorArrayV2Write/TensorListSetItemv
price_layer2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer2/while/add/y?
price_layer2/while/addAddV2price_layer2_while_placeholder!price_layer2/while/add/y:output:0*
T0*
_output_shapes
: 2
price_layer2/while/addz
price_layer2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer2/while/add_1/y?
price_layer2/while/add_1AddV22price_layer2_while_price_layer2_while_loop_counter#price_layer2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
price_layer2/while/add_1?
price_layer2/while/IdentityIdentityprice_layer2/while/add_1:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity?
price_layer2/while/Identity_1Identity8price_layer2_while_price_layer2_while_maximum_iterations6^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_1?
price_layer2/while/Identity_2Identityprice_layer2/while/add:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_2?
price_layer2/while/Identity_3IdentityGprice_layer2/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_3?
price_layer2/while/Identity_4Identity(price_layer2/while/lstm_cell_5/mul_2:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer2/while/Identity_4?
price_layer2/while/Identity_5Identity(price_layer2/while/lstm_cell_5/add_1:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer2/while/Identity_5"C
price_layer2_while_identity$price_layer2/while/Identity:output:0"G
price_layer2_while_identity_1&price_layer2/while/Identity_1:output:0"G
price_layer2_while_identity_2&price_layer2/while/Identity_2:output:0"G
price_layer2_while_identity_3&price_layer2/while/Identity_3:output:0"G
price_layer2_while_identity_4&price_layer2/while/Identity_4:output:0"G
price_layer2_while_identity_5&price_layer2/while/Identity_5:output:0"?
>price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource@price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
?price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resourceAprice_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
=price_layer2_while_lstm_cell_5_matmul_readvariableop_resource?price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0"d
/price_layer2_while_price_layer2_strided_slice_11price_layer2_while_price_layer2_strided_slice_1_0"?
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensormprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2n
5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp2l
4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp2p
6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
u
K__inference_concat_layer_layer_call_and_return_conditional_losses_595874003

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
:?????????"2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????"2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?B
?
while_body_595873379
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??(while/lstm_cell_4/BiasAdd/ReadVariableOp?'while/lstm_cell_4/MatMul/ReadVariableOp?)while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595872160

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
?Z
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595873617

inputs.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??"lstm_cell_4/BiasAdd/ReadVariableOp?!lstm_cell_4/MatMul/ReadVariableOp?#lstm_cell_4/MatMul_1/ReadVariableOp?whileD
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
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
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
while_body_595873532* 
condR
while_cond_595873531*K
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
IdentityIdentitytranspose_1:y:0#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?B
?
while_body_595873532
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??(while/lstm_cell_4/BiasAdd/ReadVariableOp?'while/lstm_cell_4/MatMul/ReadVariableOp?)while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
?[
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875167
inputs_0.
*lstm_cell_4_matmul_readvariableop_resource0
,lstm_cell_4_matmul_1_readvariableop_resource/
+lstm_cell_4_biasadd_readvariableop_resource
identity??"lstm_cell_4/BiasAdd/ReadVariableOp?!lstm_cell_4/MatMul/ReadVariableOp?#lstm_cell_4/MatMul_1/ReadVariableOp?whileF
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
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!lstm_cell_4/MatMul/ReadVariableOp?
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul?
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes

: *
dtype02%
#lstm_cell_4/MatMul_1/ReadVariableOp?
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/MatMul_1?
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/add?
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"lstm_cell_4/BiasAdd/ReadVariableOp?
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_4/BiasAddh
lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/Const|
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_4/split/split_dim?
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
lstm_cell_4/split?
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid?
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_1?
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mulz
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu?
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_1?
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/add_1?
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Sigmoid_2y
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/Relu_1?
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
lstm_cell_4/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
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
while_body_595875082* 
condR
while_cond_595875081*K
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
IdentityIdentitytranspose_1:y:0#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :??????????????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?Z
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595873952

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595873867* 
condR
while_cond_595873866*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
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
:????????? *
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
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_price_layer1_layer_call_fn_595875670

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
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958736172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
while_body_595872619
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_4_595872643_0!
while_lstm_cell_4_595872645_0!
while_lstm_cell_4_595872647_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_4_595872643
while_lstm_cell_4_595872645
while_lstm_cell_4_595872647??)while/lstm_cell_4/StatefulPartitionedCall?
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
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_595872643_0while_lstm_cell_4_595872645_0while_lstm_cell_4_595872647_0*
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
GPU 2J 8? *S
fNRL
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_5958721932+
)while/lstm_cell_4/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_4/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_4/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2*^while/lstm_cell_4/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_4_595872643while_lstm_cell_4_595872643_0"<
while_lstm_cell_4_595872645while_lstm_cell_4_595872645_0"<
while_lstm_cell_4_595872647while_lstm_cell_4_595872647_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 
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
?B
?
while_body_595875410
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_06
2while_lstm_cell_4_matmul_readvariableop_resource_08
4while_lstm_cell_4_matmul_1_readvariableop_resource_07
3while_lstm_cell_4_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor4
0while_lstm_cell_4_matmul_readvariableop_resource6
2while_lstm_cell_4_matmul_1_readvariableop_resource5
1while_lstm_cell_4_biasadd_readvariableop_resource??(while/lstm_cell_4/BiasAdd/ReadVariableOp?'while/lstm_cell_4/MatMul/ReadVariableOp?)while/lstm_cell_4/MatMul_1/ReadVariableOp?
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
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes

: *
dtype02)
'while/lstm_cell_4/MatMul/ReadVariableOp?
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul?
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes

: *
dtype02+
)while/lstm_cell_4/MatMul_1/ReadVariableOp?
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/MatMul_1?
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/add?
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes
: *
dtype02*
(while/lstm_cell_4/BiasAdd/ReadVariableOp?
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_4/BiasAddt
while/lstm_cell_4/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_4/Const?
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_4/split/split_dim?
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:?????????:?????????:?????????:?????????*
	num_split2
while/lstm_cell_4/split?
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid?
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_1?
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul?
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu?
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_1?
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/add_1?
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Sigmoid_2?
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/Relu_1?
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:?????????2
while/lstm_cell_4/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_4/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:?????????2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :?????????:?????????: : :::2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 
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
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595876475

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
?U
?
!price_layer2_while_body_5958745186
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_35
1price_layer2_while_price_layer2_strided_slice_1_0q
mprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0C
?price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0E
Aprice_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0D
@price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0
price_layer2_while_identity!
price_layer2_while_identity_1!
price_layer2_while_identity_2!
price_layer2_while_identity_3!
price_layer2_while_identity_4!
price_layer2_while_identity_53
/price_layer2_while_price_layer2_strided_slice_1o
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensorA
=price_layer2_while_lstm_cell_5_matmul_readvariableop_resourceC
?price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resourceB
>price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource??5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp?4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp?
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6price_layer2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0price_layer2_while_placeholderMprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype028
6price_layer2/while/TensorArrayV2Read/TensorListGetItem?
4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOpReadVariableOp?price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype026
4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp?
%price_layer2/while/lstm_cell_5/MatMulMatMul=price_layer2/while/TensorArrayV2Read/TensorListGetItem:item:0<price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2'
%price_layer2/while/lstm_cell_5/MatMul?
6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOpAprice_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype028
6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp?
'price_layer2/while/lstm_cell_5/MatMul_1MatMul price_layer2_while_placeholder_2>price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'price_layer2/while/lstm_cell_5/MatMul_1?
"price_layer2/while/lstm_cell_5/addAddV2/price_layer2/while/lstm_cell_5/MatMul:product:01price_layer2/while/lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2$
"price_layer2/while/lstm_cell_5/add?
5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp@price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype027
5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp?
&price_layer2/while/lstm_cell_5/BiasAddBiasAdd&price_layer2/while/lstm_cell_5/add:z:0=price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&price_layer2/while/lstm_cell_5/BiasAdd?
$price_layer2/while/lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$price_layer2/while/lstm_cell_5/Const?
.price_layer2/while/lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :20
.price_layer2/while/lstm_cell_5/split/split_dim?
$price_layer2/while/lstm_cell_5/splitSplit7price_layer2/while/lstm_cell_5/split/split_dim:output:0/price_layer2/while/lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2&
$price_layer2/while/lstm_cell_5/split?
&price_layer2/while/lstm_cell_5/SigmoidSigmoid-price_layer2/while/lstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2(
&price_layer2/while/lstm_cell_5/Sigmoid?
(price_layer2/while/lstm_cell_5/Sigmoid_1Sigmoid-price_layer2/while/lstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2*
(price_layer2/while/lstm_cell_5/Sigmoid_1?
"price_layer2/while/lstm_cell_5/mulMul,price_layer2/while/lstm_cell_5/Sigmoid_1:y:0 price_layer2_while_placeholder_3*
T0*'
_output_shapes
:????????? 2$
"price_layer2/while/lstm_cell_5/mul?
#price_layer2/while/lstm_cell_5/ReluRelu-price_layer2/while/lstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2%
#price_layer2/while/lstm_cell_5/Relu?
$price_layer2/while/lstm_cell_5/mul_1Mul*price_layer2/while/lstm_cell_5/Sigmoid:y:01price_layer2/while/lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_5/mul_1?
$price_layer2/while/lstm_cell_5/add_1AddV2&price_layer2/while/lstm_cell_5/mul:z:0(price_layer2/while/lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_5/add_1?
(price_layer2/while/lstm_cell_5/Sigmoid_2Sigmoid-price_layer2/while/lstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2*
(price_layer2/while/lstm_cell_5/Sigmoid_2?
%price_layer2/while/lstm_cell_5/Relu_1Relu(price_layer2/while/lstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_5/Relu_1?
$price_layer2/while/lstm_cell_5/mul_2Mul,price_layer2/while/lstm_cell_5/Sigmoid_2:y:03price_layer2/while/lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_5/mul_2?
7price_layer2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer2_while_placeholder_1price_layer2_while_placeholder(price_layer2/while/lstm_cell_5/mul_2:z:0*
_output_shapes
: *
element_dtype029
7price_layer2/while/TensorArrayV2Write/TensorListSetItemv
price_layer2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer2/while/add/y?
price_layer2/while/addAddV2price_layer2_while_placeholder!price_layer2/while/add/y:output:0*
T0*
_output_shapes
: 2
price_layer2/while/addz
price_layer2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
price_layer2/while/add_1/y?
price_layer2/while/add_1AddV22price_layer2_while_price_layer2_while_loop_counter#price_layer2/while/add_1/y:output:0*
T0*
_output_shapes
: 2
price_layer2/while/add_1?
price_layer2/while/IdentityIdentityprice_layer2/while/add_1:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity?
price_layer2/while/Identity_1Identity8price_layer2_while_price_layer2_while_maximum_iterations6^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_1?
price_layer2/while/Identity_2Identityprice_layer2/while/add:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_2?
price_layer2/while/Identity_3IdentityGprice_layer2/while/TensorArrayV2Write/TensorListSetItem:output_handle:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_3?
price_layer2/while/Identity_4Identity(price_layer2/while/lstm_cell_5/mul_2:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer2/while/Identity_4?
price_layer2/while/Identity_5Identity(price_layer2/while/lstm_cell_5/add_1:z:06^price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5^price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp7^price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer2/while/Identity_5"C
price_layer2_while_identity$price_layer2/while/Identity:output:0"G
price_layer2_while_identity_1&price_layer2/while/Identity_1:output:0"G
price_layer2_while_identity_2&price_layer2/while/Identity_2:output:0"G
price_layer2_while_identity_3&price_layer2/while/Identity_3:output:0"G
price_layer2_while_identity_4&price_layer2/while/Identity_4:output:0"G
price_layer2_while_identity_5&price_layer2/while/Identity_5:output:0"?
>price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource@price_layer2_while_lstm_cell_5_biasadd_readvariableop_resource_0"?
?price_layer2_while_lstm_cell_5_matmul_1_readvariableop_resourceAprice_layer2_while_lstm_cell_5_matmul_1_readvariableop_resource_0"?
=price_layer2_while_lstm_cell_5_matmul_readvariableop_resource?price_layer2_while_lstm_cell_5_matmul_readvariableop_resource_0"d
/price_layer2_while_price_layer2_strided_slice_11price_layer2_while_price_layer2_strided_slice_1_0"?
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensormprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2n
5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp5price_layer2/while/lstm_cell_5/BiasAdd/ReadVariableOp2l
4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp4price_layer2/while/lstm_cell_5/MatMul/ReadVariableOp2p
6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp6price_layer2/while/lstm_cell_5/MatMul_1/ReadVariableOp: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?Z
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595876304

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595876219* 
condR
while_cond_595876218*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
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
:????????? *
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
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
while_body_595873229
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_5_595873253_0!
while_lstm_cell_5_595873255_0!
while_lstm_cell_5_595873257_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5_595873253
while_lstm_cell_5_595873255
while_lstm_cell_5_595873257??)while/lstm_cell_5/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_595873253_0while_lstm_cell_5_595873255_0while_lstm_cell_5_595873257_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_5958728032+
)while/lstm_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_5_595873253while_lstm_cell_5_595873253_0"<
while_lstm_cell_5_595873255while_lstm_cell_5_595873255_0"<
while_lstm_cell_5_595873257while_lstm_cell_5_595873257_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_595875234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595875234___redundant_placeholder07
3while_while_cond_595875234___redundant_placeholder17
3while_while_cond_595875234___redundant_placeholder27
3while_while_cond_595875234___redundant_placeholder3
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
?Z
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595876151

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595876066* 
condR
while_cond_595876065*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
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
:????????? *
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
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_595876400

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
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595876542

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
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?
?
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595876575

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
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
add?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
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
L:????????? :????????? :????????? :????????? *
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:????????? 2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:????????? 2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:????????? 2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:????????? 2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:????????? 2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:????????? 2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:????????? 2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:????????? 2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
mul_2?
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_1?

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/0:QM
'
_output_shapes
:????????? 
"
_user_specified_name
states/1
?&
?
F__inference_model_4_layer_call_and_return_conditional_losses_595874235

inputs
inputs_1
price_layer1_595874203
price_layer1_595874205
price_layer1_595874207
price_layer2_595874210
price_layer2_595874212
price_layer2_595874214
fixed_layer1_595874219
fixed_layer1_595874221
fixed_layer2_595874224
fixed_layer2_595874226
action_output_595874229
action_output_595874231
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_595874203price_layer1_595874205price_layer1_595874207*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958736172&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_595874210price_layer2_595874212price_layer2_595874214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958739522&
$price_layer2/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_5958739882
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_5958740032
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_595874219fixed_layer1_595874221*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_5958740232&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_595874224fixed_layer2_595874226*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_5958740502&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_595874229action_output_595874231*
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
L__inference_action_output_layer_call_and_return_conditional_losses_5958740762'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2N
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
?Z
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595873799

inputs.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileD
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
:?????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595873714* 
condR
while_cond_595873713*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
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
:????????? *
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
:????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_model_4_layer_call_fn_595874984
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
	unknown_8
	unknown_9

unknown_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_4_layer_call_and_return_conditional_losses_5958741692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::22
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
?[
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595875823
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileF
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595875738* 
condR
while_cond_595875737*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
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
:????????? *
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
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595876442

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
?%
?
while_body_595873097
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0!
while_lstm_cell_5_595873121_0!
while_lstm_cell_5_595873123_0!
while_lstm_cell_5_595873125_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_5_595873121
while_lstm_cell_5_595873123
while_lstm_cell_5_595873125??)while/lstm_cell_5/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
)while/lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_5_595873121_0while_lstm_cell_5_595873123_0while_lstm_cell_5_595873125_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_5958727702+
)while/lstm_cell_5/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_5/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_5/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity2while/lstm_cell_5/StatefulPartitionedCall:output:1*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity2while/lstm_cell_5/StatefulPartitionedCall:output:2*^while/lstm_cell_5/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"<
while_lstm_cell_5_595873121while_lstm_cell_5_595873121_0"<
while_lstm_cell_5_595873123while_lstm_cell_5_595873123_0"<
while_lstm_cell_5_595873125while_lstm_cell_5_595873125_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_5/StatefulPartitionedCall)while/lstm_cell_5/StatefulPartitionedCall: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_595875737
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595875737___redundant_placeholder07
3while_while_cond_595875737___redundant_placeholder17
3while_while_cond_595875737___redundant_placeholder27
3while_while_cond_595875737___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
!price_layer2_while_cond_5958745176
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_38
4price_layer2_while_less_price_layer2_strided_slice_1Q
Mprice_layer2_while_price_layer2_while_cond_595874517___redundant_placeholder0Q
Mprice_layer2_while_price_layer2_while_cond_595874517___redundant_placeholder1Q
Mprice_layer2_while_price_layer2_while_cond_595874517___redundant_placeholder2Q
Mprice_layer2_while_price_layer2_while_cond_595874517___redundant_placeholder3
price_layer2_while_identity
?
price_layer2/while/LessLessprice_layer2_while_placeholder4price_layer2_while_less_price_layer2_strided_slice_1*
T0*
_output_shapes
: 2
price_layer2/while/Less?
price_layer2/while/IdentityIdentityprice_layer2/while/Less:z:0*
T0
*
_output_shapes
: 2
price_layer2/while/Identity"C
price_layer2_while_identity$price_layer2/while/Identity:output:0*S
_input_shapesB
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?&
?
F__inference_model_4_layer_call_and_return_conditional_losses_595874129
price_input
	env_input
price_layer1_595874097
price_layer1_595874099
price_layer1_595874101
price_layer2_595874104
price_layer2_595874106
price_layer2_595874108
fixed_layer1_595874113
fixed_layer1_595874115
fixed_layer2_595874118
fixed_layer2_595874120
action_output_595874123
action_output_595874125
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_595874097price_layer1_595874099price_layer1_595874101*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958736172&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_595874104price_layer2_595874106price_layer2_595874108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958739522&
$price_layer2/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_5958739882
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0	env_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_5958740032
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_595874113fixed_layer1_595874115*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_5958740232&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_595874118fixed_layer2_595874120*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_5958740502&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_595874123action_output_595874125*
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
L__inference_action_output_layer_call_and_return_conditional_losses_5958740762'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2N
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
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595872193

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
?
?
)model_4_price_layer1_while_cond_595871828F
Bmodel_4_price_layer1_while_model_4_price_layer1_while_loop_counterL
Hmodel_4_price_layer1_while_model_4_price_layer1_while_maximum_iterations*
&model_4_price_layer1_while_placeholder,
(model_4_price_layer1_while_placeholder_1,
(model_4_price_layer1_while_placeholder_2,
(model_4_price_layer1_while_placeholder_3H
Dmodel_4_price_layer1_while_less_model_4_price_layer1_strided_slice_1a
]model_4_price_layer1_while_model_4_price_layer1_while_cond_595871828___redundant_placeholder0a
]model_4_price_layer1_while_model_4_price_layer1_while_cond_595871828___redundant_placeholder1a
]model_4_price_layer1_while_model_4_price_layer1_while_cond_595871828___redundant_placeholder2a
]model_4_price_layer1_while_model_4_price_layer1_while_cond_595871828___redundant_placeholder3'
#model_4_price_layer1_while_identity
?
model_4/price_layer1/while/LessLess&model_4_price_layer1_while_placeholderDmodel_4_price_layer1_while_less_model_4_price_layer1_strided_slice_1*
T0*
_output_shapes
: 2!
model_4/price_layer1/while/Less?
#model_4/price_layer1/while/IdentityIdentity#model_4/price_layer1/while/Less:z:0*
T0
*
_output_shapes
: 2%
#model_4/price_layer1/while/Identity"S
#model_4_price_layer1_while_identity,model_4/price_layer1/while/Identity:output:0*S
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
?
?
while_cond_595873096
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595873096___redundant_placeholder07
3while_while_cond_595873096___redundant_placeholder17
3while_while_cond_595873096___redundant_placeholder27
3while_while_cond_595873096___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
0__inference_price_layer1_layer_call_fn_595875342
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
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958726882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

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
?&
?
F__inference_model_4_layer_call_and_return_conditional_losses_595874169

inputs
inputs_1
price_layer1_595874137
price_layer1_595874139
price_layer1_595874141
price_layer2_595874144
price_layer2_595874146
price_layer2_595874148
fixed_layer1_595874153
fixed_layer1_595874155
fixed_layer2_595874158
fixed_layer2_595874160
action_output_595874163
action_output_595874165
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_595874137price_layer1_595874139price_layer1_595874141*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958734642&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_595874144price_layer2_595874146price_layer2_595874148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958737992&
$price_layer2/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_5958739882
price_flatten/PartitionedCall?
concat_layer/PartitionedCallPartitionedCall&price_flatten/PartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????"* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_concat_layer_layer_call_and_return_conditional_losses_5958740032
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_595874153fixed_layer1_595874155*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_5958740232&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_595874158fixed_layer2_595874160*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_5958740502&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_595874163action_output_595874165*
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
L__inference_action_output_layer_call_and_return_conditional_losses_5958740762'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2N
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
?
?
while_cond_595876218
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595876218___redundant_placeholder07
3while_while_cond_595876218___redundant_placeholder17
3while_while_cond_595876218___redundant_placeholder27
3while_while_cond_595876218___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
while_cond_595875081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595875081___redundant_placeholder07
3while_while_cond_595875081___redundant_placeholder17
3while_while_cond_595875081___redundant_placeholder27
3while_while_cond_595875081___redundant_placeholder3
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
?D
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595873298

inputs
lstm_cell_5_595873216
lstm_cell_5_595873218
lstm_cell_5_595873220
identity??#lstm_cell_5/StatefulPartitionedCall?whileD
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
#lstm_cell_5/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_5_595873216lstm_cell_5_595873218lstm_cell_5_595873220*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:????????? :????????? :????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_5958728032%
#lstm_cell_5/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_5_595873216lstm_cell_5_595873218lstm_cell_5_595873220*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595873229* 
condR
while_cond_595873228*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
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
:????????? *
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
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_5/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_5/StatefulPartitionedCall#lstm_cell_5/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?[
?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595875976
inputs_0.
*lstm_cell_5_matmul_readvariableop_resource0
,lstm_cell_5_matmul_1_readvariableop_resource/
+lstm_cell_5_biasadd_readvariableop_resource
identity??"lstm_cell_5/BiasAdd/ReadVariableOp?!lstm_cell_5/MatMul/ReadVariableOp?#lstm_cell_5/MatMul_1/ReadVariableOp?whileF
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
value	B : 2
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
value	B : 2
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
:????????? 2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2	
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
 :??????????????????2
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
valueB"????   27
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
:?????????*
shrink_axis_mask2
strided_slice_2?
!lstm_cell_5/MatMul/ReadVariableOpReadVariableOp*lstm_cell_5_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02#
!lstm_cell_5/MatMul/ReadVariableOp?
lstm_cell_5/MatMulMatMulstrided_slice_2:output:0)lstm_cell_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul?
#lstm_cell_5/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_5_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02%
#lstm_cell_5/MatMul_1/ReadVariableOp?
lstm_cell_5/MatMul_1MatMulzeros:output:0+lstm_cell_5/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/MatMul_1?
lstm_cell_5/addAddV2lstm_cell_5/MatMul:product:0lstm_cell_5/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/add?
"lstm_cell_5/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02$
"lstm_cell_5/BiasAdd/ReadVariableOp?
lstm_cell_5/BiasAddBiasAddlstm_cell_5/add:z:0*lstm_cell_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_5/BiasAddh
lstm_cell_5/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/Const|
lstm_cell_5/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_5/split/split_dim?
lstm_cell_5/splitSplit$lstm_cell_5/split/split_dim:output:0lstm_cell_5/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_5/split?
lstm_cell_5/SigmoidSigmoidlstm_cell_5/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid?
lstm_cell_5/Sigmoid_1Sigmoidlstm_cell_5/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_1?
lstm_cell_5/mulMullstm_cell_5/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mulz
lstm_cell_5/ReluRelulstm_cell_5/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu?
lstm_cell_5/mul_1Mullstm_cell_5/Sigmoid:y:0lstm_cell_5/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_1?
lstm_cell_5/add_1AddV2lstm_cell_5/mul:z:0lstm_cell_5/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/add_1?
lstm_cell_5/Sigmoid_2Sigmoidlstm_cell_5/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Sigmoid_2y
lstm_cell_5/Relu_1Relulstm_cell_5/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/Relu_1?
lstm_cell_5/mul_2Mullstm_cell_5/Sigmoid_2:y:0 lstm_cell_5/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_5/mul_2?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_5_matmul_readvariableop_resource,lstm_cell_5_matmul_1_readvariableop_resource+lstm_cell_5_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
* 
bodyR
while_body_595875891* 
condR
while_cond_595875890*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :?????????????????? *
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
:????????? *
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
 :?????????????????? 2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0#^lstm_cell_5/BiasAdd/ReadVariableOp"^lstm_cell_5/MatMul/ReadVariableOp$^lstm_cell_5/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2H
"lstm_cell_5/BiasAdd/ReadVariableOp"lstm_cell_5/BiasAdd/ReadVariableOp2F
!lstm_cell_5/MatMul/ReadVariableOp!lstm_cell_5/MatMul/ReadVariableOp2J
#lstm_cell_5/MatMul_1/ReadVariableOp#lstm_cell_5/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?
?
while_cond_595876065
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_17
3while_while_cond_595876065___redundant_placeholder07
3while_while_cond_595876065___redundant_placeholder17
3while_while_cond_595876065___redundant_placeholder27
3while_while_cond_595876065___redundant_placeholder3
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
@: : : : :????????? :????????? : ::::: 
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
:????????? :-)
'
_output_shapes
:????????? :

_output_shapes
: :

_output_shapes
:
?
?
0__inference_price_layer1_layer_call_fn_595875659

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
 *+
_output_shapes
:?????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958734642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_price_flatten_layer_call_fn_595876337

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
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_flatten_layer_call_and_return_conditional_losses_5958739882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*&
_input_shapes
:????????? :O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
0__inference_price_layer1_layer_call_fn_595875331
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
 *4
_output_shapes"
 :??????????????????*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_5958725562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :??????????????????2

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
0__inference_price_layer2_layer_call_fn_595876326

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
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer2_layer_call_and_return_conditional_losses_5958739522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_fixed_layer2_layer_call_fn_595876390

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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_5958740502
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
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
loss
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?L
_tf_keras_network?L{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5, 1]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}}, "training_config": {"loss": {"action_output": "mse"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "price_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}}
?
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "price_layer1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
?
cell

state_spec
regularization_losses
trainable_variables
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "price_layer2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 8]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 8]}}
?
regularization_losses
	variables
trainable_variables
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "price_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "env_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}}
?
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 2]}]}
?

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34]}}
?

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

1kernel
2bias
3regularization_losses
4	variables
5trainable_variables
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "action_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
I
7iter
	8decay
9learning_rate
:momentum"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
;0
<1
=2
>3
?4
@5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
v
;0
<1
=2
>3
?4
@5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
?
Ametrics
Blayer_regularization_losses

Clayers
Dnon_trainable_variables
regularization_losses
	variables
Elayer_metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?

;kernel
<recurrent_kernel
=bias
Fregularization_losses
G	variables
Htrainable_variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_4", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
?
Jmetrics
Klayer_regularization_losses

Llayers
Mnon_trainable_variables
regularization_losses
trainable_variables
	variables
Nlayer_metrics

Ostates
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

>kernel
?recurrent_kernel
@bias
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_5", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
?
Tmetrics
Ulayer_regularization_losses

Vlayers
Wnon_trainable_variables
regularization_losses
trainable_variables
	variables
Xlayer_metrics

Ystates
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Zlayer_regularization_losses
[metrics

\layers
]non_trainable_variables
regularization_losses
	variables
^layer_metrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
_layer_regularization_losses
`metrics

alayers
bnon_trainable_variables
!regularization_losses
"	variables
clayer_metrics
#trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#"2fixed_layer1/kernel
:2fixed_layer1/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
dlayer_regularization_losses
emetrics

flayers
gnon_trainable_variables
'regularization_losses
(	variables
hlayer_metrics
)trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2fixed_layer2/kernel
:2fixed_layer2/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
ilayer_regularization_losses
jmetrics

klayers
lnon_trainable_variables
-regularization_losses
.	variables
mlayer_metrics
/trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2action_output/kernel
 :2action_output/bias
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
nlayer_regularization_losses
ometrics

players
qnon_trainable_variables
3regularization_losses
4	variables
rlayer_metrics
5trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
1:/ 2price_layer1/lstm_cell_4/kernel
;:9 2)price_layer1/lstm_cell_4/recurrent_kernel
+:) 2price_layer1/lstm_cell_4/bias
2:0	?2price_layer2/lstm_cell_5/kernel
<::	 ?2)price_layer2/lstm_cell_5/recurrent_kernel
,:*?2price_layer2/lstm_cell_5/bias
'
s0"
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
5
;0
<1
=2"
trackable_list_wrapper
?
tlayer_regularization_losses
umetrics

vlayers
wnon_trainable_variables
Fregularization_losses
G	variables
xlayer_metrics
Htrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
5
>0
?1
@2"
trackable_list_wrapper
?
ylayer_regularization_losses
zmetrics

{layers
|non_trainable_variables
Pregularization_losses
Q	variables
}layer_metrics
Rtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
	~total
	count
?	variables
?	keras_api"?
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
~0
1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
F__inference_model_4_layer_call_and_return_conditional_losses_595874129
F__inference_model_4_layer_call_and_return_conditional_losses_595874093
F__inference_model_4_layer_call_and_return_conditional_losses_595874954
F__inference_model_4_layer_call_and_return_conditional_losses_595874627?
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
$__inference__wrapped_model_595872087?
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
+__inference_model_4_layer_call_fn_595874196
+__inference_model_4_layer_call_fn_595874984
+__inference_model_4_layer_call_fn_595874262
+__inference_model_4_layer_call_fn_595875014?
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
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875495
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875648
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875167
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875320?
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
0__inference_price_layer1_layer_call_fn_595875331
0__inference_price_layer1_layer_call_fn_595875659
0__inference_price_layer1_layer_call_fn_595875670
0__inference_price_layer1_layer_call_fn_595875342?
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
?2?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595875823
K__inference_price_layer2_layer_call_and_return_conditional_losses_595876151
K__inference_price_layer2_layer_call_and_return_conditional_losses_595875976
K__inference_price_layer2_layer_call_and_return_conditional_losses_595876304?
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
0__inference_price_layer2_layer_call_fn_595876326
0__inference_price_layer2_layer_call_fn_595876315
0__inference_price_layer2_layer_call_fn_595875987
0__inference_price_layer2_layer_call_fn_595875998?
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_595876332?
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
1__inference_price_flatten_layer_call_fn_595876337?
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_595876344?
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
0__inference_concat_layer_layer_call_fn_595876350?
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_595876361?
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
0__inference_fixed_layer1_layer_call_fn_595876370?
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_595876381?
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
0__inference_fixed_layer2_layer_call_fn_595876390?
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
L__inference_action_output_layer_call_and_return_conditional_losses_595876400?
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
1__inference_action_output_layer_call_fn_595876409?
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
'__inference_signature_wrapper_595874300	env_inputprice_input"?
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
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595876475
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595876442?
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
/__inference_lstm_cell_4_layer_call_fn_595876509
/__inference_lstm_cell_4_layer_call_fn_595876492?
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
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595876542
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595876575?
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
/__inference_lstm_cell_5_layer_call_fn_595876609
/__inference_lstm_cell_5_layer_call_fn_595876592?
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
$__inference__wrapped_model_595872087?;<=>?@%&+,12b?_
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
L__inference_action_output_layer_call_and_return_conditional_losses_595876400\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_action_output_layer_call_fn_595876409O12/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_concat_layer_layer_call_and_return_conditional_losses_595876344?Z?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "%?"
?
0?????????"
? ?
0__inference_concat_layer_layer_call_fn_595876350vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????"?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_595876361\%&/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????
? ?
0__inference_fixed_layer1_layer_call_fn_595876370O%&/?,
%?"
 ?
inputs?????????"
? "???????????
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_595876381\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
0__inference_fixed_layer2_layer_call_fn_595876390O+,/?,
%?"
 ?
inputs?????????
? "???????????
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595876442?;<=??}
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
J__inference_lstm_cell_4_layer_call_and_return_conditional_losses_595876475?;<=??}
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
/__inference_lstm_cell_4_layer_call_fn_595876492?;<=??}
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
/__inference_lstm_cell_4_layer_call_fn_595876509?;<=??}
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
1/1??????????
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595876542?>?@??}
v?s
 ?
inputs?????????
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
J__inference_lstm_cell_5_layer_call_and_return_conditional_losses_595876575?>?@??}
v?s
 ?
inputs?????????
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "s?p
i?f
?
0/0????????? 
E?B
?
0/1/0????????? 
?
0/1/1????????? 
? ?
/__inference_lstm_cell_5_layer_call_fn_595876592?>?@??}
v?s
 ?
inputs?????????
K?H
"?
states/0????????? 
"?
states/1????????? 
p
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
/__inference_lstm_cell_5_layer_call_fn_595876609?>?@??}
v?s
 ?
inputs?????????
K?H
"?
states/0????????? 
"?
states/1????????? 
p 
? "c?`
?
0????????? 
A?>
?
1/0????????? 
?
1/1????????? ?
F__inference_model_4_layer_call_and_return_conditional_losses_595874093?;<=>?@%&+,12j?g
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
F__inference_model_4_layer_call_and_return_conditional_losses_595874129?;<=>?@%&+,12j?g
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
F__inference_model_4_layer_call_and_return_conditional_losses_595874627?;<=>?@%&+,12f?c
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
F__inference_model_4_layer_call_and_return_conditional_losses_595874954?;<=>?@%&+,12f?c
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
+__inference_model_4_layer_call_fn_595874196?;<=>?@%&+,12j?g
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
+__inference_model_4_layer_call_fn_595874262?;<=>?@%&+,12j?g
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
+__inference_model_4_layer_call_fn_595874984?;<=>?@%&+,12f?c
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
+__inference_model_4_layer_call_fn_595875014?;<=>?@%&+,12f?c
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_595876332X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
1__inference_price_flatten_layer_call_fn_595876337K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875167?;<=O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "2?/
(?%
0??????????????????
? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875320?;<=O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "2?/
(?%
0??????????????????
? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875495q;<=??<
5?2
$?!
inputs?????????

 
p

 
? ")?&
?
0?????????
? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_595875648q;<=??<
5?2
$?!
inputs?????????

 
p 

 
? ")?&
?
0?????????
? ?
0__inference_price_layer1_layer_call_fn_595875331};<=O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"???????????????????
0__inference_price_layer1_layer_call_fn_595875342};<=O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"???????????????????
0__inference_price_layer1_layer_call_fn_595875659d;<=??<
5?2
$?!
inputs?????????

 
p

 
? "???????????
0__inference_price_layer1_layer_call_fn_595875670d;<=??<
5?2
$?!
inputs?????????

 
p 

 
? "???????????
K__inference_price_layer2_layer_call_and_return_conditional_losses_595875823}>?@O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"
?
0????????? 
? ?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595875976}>?@O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"
?
0????????? 
? ?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595876151m>?@??<
5?2
$?!
inputs?????????

 
p

 
? "%?"
?
0????????? 
? ?
K__inference_price_layer2_layer_call_and_return_conditional_losses_595876304m>?@??<
5?2
$?!
inputs?????????

 
p 

 
? "%?"
?
0????????? 
? ?
0__inference_price_layer2_layer_call_fn_595875987p>?@O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "?????????? ?
0__inference_price_layer2_layer_call_fn_595875998p>?@O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "?????????? ?
0__inference_price_layer2_layer_call_fn_595876315`>?@??<
5?2
$?!
inputs?????????

 
p

 
? "?????????? ?
0__inference_price_layer2_layer_call_fn_595876326`>?@??<
5?2
$?!
inputs?????????

 
p 

 
? "?????????? ?
'__inference_signature_wrapper_595874300?;<=>?@%&+,12y?v
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