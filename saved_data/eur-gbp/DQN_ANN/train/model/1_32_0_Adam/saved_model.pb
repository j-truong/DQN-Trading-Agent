ԏ
??
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
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8Ң
?
price_layer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameprice_layer1/kernel
{
'price_layer1/kernel/Read/ReadVariableOpReadVariableOpprice_layer1/kernel*
_output_shapes

: *
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
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
?
Adam/price_layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/price_layer1/kernel/m
?
.Adam/price_layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/price_layer1/kernel/m*
_output_shapes

: *
dtype0
?
Adam/price_layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/price_layer1/bias/m
?
,Adam/price_layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/price_layer1/bias/m*
_output_shapes
: *
dtype0
?
Adam/fixed_layer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"*+
shared_nameAdam/fixed_layer1/kernel/m
?
.Adam/fixed_layer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fixed_layer1/kernel/m*
_output_shapes

:"*
dtype0
?
Adam/fixed_layer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/fixed_layer1/bias/m
?
,Adam/fixed_layer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/fixed_layer1/bias/m*
_output_shapes
:*
dtype0
?
Adam/fixed_layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/fixed_layer2/kernel/m
?
.Adam/fixed_layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/fixed_layer2/kernel/m*
_output_shapes

:*
dtype0
?
Adam/fixed_layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/fixed_layer2/bias/m
?
,Adam/fixed_layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/fixed_layer2/bias/m*
_output_shapes
:*
dtype0
?
Adam/action_output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameAdam/action_output/kernel/m
?
/Adam/action_output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/action_output/kernel/m*
_output_shapes

:*
dtype0
?
Adam/action_output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/action_output/bias/m
?
-Adam/action_output/bias/m/Read/ReadVariableOpReadVariableOpAdam/action_output/bias/m*
_output_shapes
:*
dtype0
?
Adam/price_layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *+
shared_nameAdam/price_layer1/kernel/v
?
.Adam/price_layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/price_layer1/kernel/v*
_output_shapes

: *
dtype0
?
Adam/price_layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/price_layer1/bias/v
?
,Adam/price_layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/price_layer1/bias/v*
_output_shapes
: *
dtype0
?
Adam/fixed_layer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:"*+
shared_nameAdam/fixed_layer1/kernel/v
?
.Adam/fixed_layer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fixed_layer1/kernel/v*
_output_shapes

:"*
dtype0
?
Adam/fixed_layer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/fixed_layer1/bias/v
?
,Adam/fixed_layer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/fixed_layer1/bias/v*
_output_shapes
:*
dtype0
?
Adam/fixed_layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameAdam/fixed_layer2/kernel/v
?
.Adam/fixed_layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/fixed_layer2/kernel/v*
_output_shapes

:*
dtype0
?
Adam/fixed_layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/fixed_layer2/bias/v
?
,Adam/fixed_layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/fixed_layer2/bias/v*
_output_shapes
:*
dtype0
?
Adam/action_output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameAdam/action_output/kernel/v
?
/Adam/action_output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/action_output/kernel/v*
_output_shapes

:*
dtype0
?
Adam/action_output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameAdam/action_output/bias/v
?
-Adam/action_output/bias/v/Read/ReadVariableOpReadVariableOpAdam/action_output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?1
value?1B?1 B?1
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
 
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
?
0iter

1beta_1

2beta_2
	3decay
4learning_ratem]m^m_m`$ma%mb*mc+mdvevfvgvh$vi%vj*vk+vl
 
8
0
1
2
3
$4
%5
*6
+7
8
0
1
2
3
$4
%5
*6
+7
 
?

5layers
	variables
6layer_metrics
7layer_regularization_losses
trainable_variables
8non_trainable_variables
9metrics
regularization_losses
 
_]
VARIABLE_VALUEprice_layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

:layers
	variables
;layer_metrics
<layer_regularization_losses
trainable_variables
=non_trainable_variables
>metrics
regularization_losses
 
 
 
?

?layers
	variables
@layer_metrics
Alayer_regularization_losses
trainable_variables
Bnon_trainable_variables
Cmetrics
regularization_losses
 
 
 
?

Dlayers
	variables
Elayer_metrics
Flayer_regularization_losses
trainable_variables
Gnon_trainable_variables
Hmetrics
regularization_losses
_]
VARIABLE_VALUEfixed_layer1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

Ilayers
 	variables
Jlayer_metrics
Klayer_regularization_losses
!trainable_variables
Lnon_trainable_variables
Mmetrics
"regularization_losses
_]
VARIABLE_VALUEfixed_layer2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?

Nlayers
&	variables
Olayer_metrics
Player_regularization_losses
'trainable_variables
Qnon_trainable_variables
Rmetrics
(regularization_losses
`^
VARIABLE_VALUEaction_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEaction_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?

Slayers
,	variables
Tlayer_metrics
Ulayer_regularization_losses
-trainable_variables
Vnon_trainable_variables
Wmetrics
.regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
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

X0
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
	Ytotal
	Zcount
[	variables
\	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

[	variables
??
VARIABLE_VALUEAdam/price_layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/price_layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/action_output/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/action_output/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/price_layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/price_layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/action_output/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/action_output/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_env_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
~
serving_default_price_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_env_inputserving_default_price_inputprice_layer1/kernelprice_layer1/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *0
f+R)
'__inference_signature_wrapper_475701662
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'price_layer1/kernel/Read/ReadVariableOp%price_layer1/bias/Read/ReadVariableOp'fixed_layer1/kernel/Read/ReadVariableOp%fixed_layer1/bias/Read/ReadVariableOp'fixed_layer2/kernel/Read/ReadVariableOp%fixed_layer2/bias/Read/ReadVariableOp(action_output/kernel/Read/ReadVariableOp&action_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/price_layer1/kernel/m/Read/ReadVariableOp,Adam/price_layer1/bias/m/Read/ReadVariableOp.Adam/fixed_layer1/kernel/m/Read/ReadVariableOp,Adam/fixed_layer1/bias/m/Read/ReadVariableOp.Adam/fixed_layer2/kernel/m/Read/ReadVariableOp,Adam/fixed_layer2/bias/m/Read/ReadVariableOp/Adam/action_output/kernel/m/Read/ReadVariableOp-Adam/action_output/bias/m/Read/ReadVariableOp.Adam/price_layer1/kernel/v/Read/ReadVariableOp,Adam/price_layer1/bias/v/Read/ReadVariableOp.Adam/fixed_layer1/kernel/v/Read/ReadVariableOp,Adam/fixed_layer1/bias/v/Read/ReadVariableOp.Adam/fixed_layer2/kernel/v/Read/ReadVariableOp,Adam/fixed_layer2/bias/v/Read/ReadVariableOp/Adam/action_output/kernel/v/Read/ReadVariableOp-Adam/action_output/bias/v/Read/ReadVariableOpConst*,
Tin%
#2!	*
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
"__inference__traced_save_475701998
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameprice_layer1/kernelprice_layer1/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/price_layer1/kernel/mAdam/price_layer1/bias/mAdam/fixed_layer1/kernel/mAdam/fixed_layer1/bias/mAdam/fixed_layer2/kernel/mAdam/fixed_layer2/bias/mAdam/action_output/kernel/mAdam/action_output/bias/mAdam/price_layer1/kernel/vAdam/price_layer1/bias/vAdam/fixed_layer1/kernel/vAdam/fixed_layer1/bias/vAdam/fixed_layer2/kernel/vAdam/fixed_layer2/bias/vAdam/action_output/kernel/vAdam/action_output/bias/v*+
Tin$
"2 *
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
%__inference__traced_restore_475702101ӗ
?
w
K__inference_concat_layer_layer_call_and_return_conditional_losses_475701816
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
?
?
F__inference_model_3_layer_call_and_return_conditional_losses_475701531
price_input
	env_input
price_layer1_475701508
price_layer1_475701510
fixed_layer1_475701515
fixed_layer1_475701517
fixed_layer2_475701520
fixed_layer2_475701522
action_output_475701525
action_output_475701527
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_475701508price_layer1_475701510*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_4757013772&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_4757013992
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_4757014142
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_475701515fixed_layer1_475701517*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_4757014342&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_475701520fixed_layer2_475701522*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_4757014612&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_475701525action_output_475701527*
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
L__inference_action_output_layer_call_and_return_conditional_losses_4757014872'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?
?
F__inference_model_3_layer_call_and_return_conditional_losses_475701504
price_input
	env_input
price_layer1_475701388
price_layer1_475701390
fixed_layer1_475701445
fixed_layer1_475701447
fixed_layer2_475701472
fixed_layer2_475701474
action_output_475701498
action_output_475701500
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_475701388price_layer1_475701390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_4757013772&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_4757013992
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_4757014142
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_475701445fixed_layer1_475701447*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_4757014342&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_475701472fixed_layer2_475701474*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_4757014612&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_475701498action_output_475701500*
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
L__inference_action_output_layer_call_and_return_conditional_losses_4757014872'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?
?
0__inference_fixed_layer1_layer_call_fn_475701842

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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_4757014342
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
??
?
%__inference__traced_restore_475702101
file_prefix(
$assignvariableop_price_layer1_kernel(
$assignvariableop_1_price_layer1_bias*
&assignvariableop_2_fixed_layer1_kernel(
$assignvariableop_3_fixed_layer1_bias*
&assignvariableop_4_fixed_layer2_kernel(
$assignvariableop_5_fixed_layer2_bias+
'assignvariableop_6_action_output_kernel)
%assignvariableop_7_action_output_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count2
.assignvariableop_15_adam_price_layer1_kernel_m0
,assignvariableop_16_adam_price_layer1_bias_m2
.assignvariableop_17_adam_fixed_layer1_kernel_m0
,assignvariableop_18_adam_fixed_layer1_bias_m2
.assignvariableop_19_adam_fixed_layer2_kernel_m0
,assignvariableop_20_adam_fixed_layer2_bias_m3
/assignvariableop_21_adam_action_output_kernel_m1
-assignvariableop_22_adam_action_output_bias_m2
.assignvariableop_23_adam_price_layer1_kernel_v0
,assignvariableop_24_adam_price_layer1_bias_v2
.assignvariableop_25_adam_fixed_layer1_kernel_v0
,assignvariableop_26_adam_fixed_layer1_bias_v2
.assignvariableop_27_adam_fixed_layer2_kernel_v0
,assignvariableop_28_adam_fixed_layer2_bias_v3
/assignvariableop_29_adam_action_output_kernel_v1
-assignvariableop_30_adam_action_output_bias_v
identity_32??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::*.
dtypes$
"2 	2
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
AssignVariableOp_2AssignVariableOp&assignvariableop_2_fixed_layer1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_fixed_layer1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_fixed_layer2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_fixed_layer2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp'assignvariableop_6_action_output_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp%assignvariableop_7_action_output_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_adam_price_layer1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_adam_price_layer1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_fixed_layer1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_fixed_layer1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_fixed_layer2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_fixed_layer2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_adam_action_output_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp-assignvariableop_22_adam_action_output_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_price_layer1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_price_layer1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_fixed_layer1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_fixed_layer1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_fixed_layer2_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_fixed_layer2_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_action_output_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_adam_action_output_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_309
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_31Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_31?
Identity_32IdentityIdentity_31:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_32"#
identity_32Identity_32:output:0*?
_input_shapes?
~: :::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302(
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
?
u
K__inference_concat_layer_layer_call_and_return_conditional_losses_475701414

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
?
?
1__inference_action_output_layer_call_fn_475701881

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
L__inference_action_output_layer_call_and_return_conditional_losses_4757014872
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
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_475701399

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
?F
?
"__inference__traced_save_475701998
file_prefix2
.savev2_price_layer1_kernel_read_readvariableop0
,savev2_price_layer1_bias_read_readvariableop2
.savev2_fixed_layer1_kernel_read_readvariableop0
,savev2_fixed_layer1_bias_read_readvariableop2
.savev2_fixed_layer2_kernel_read_readvariableop0
,savev2_fixed_layer2_bias_read_readvariableop3
/savev2_action_output_kernel_read_readvariableop1
-savev2_action_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_price_layer1_kernel_m_read_readvariableop7
3savev2_adam_price_layer1_bias_m_read_readvariableop9
5savev2_adam_fixed_layer1_kernel_m_read_readvariableop7
3savev2_adam_fixed_layer1_bias_m_read_readvariableop9
5savev2_adam_fixed_layer2_kernel_m_read_readvariableop7
3savev2_adam_fixed_layer2_bias_m_read_readvariableop:
6savev2_adam_action_output_kernel_m_read_readvariableop8
4savev2_adam_action_output_bias_m_read_readvariableop9
5savev2_adam_price_layer1_kernel_v_read_readvariableop7
3savev2_adam_price_layer1_bias_v_read_readvariableop9
5savev2_adam_fixed_layer1_kernel_v_read_readvariableop7
3savev2_adam_fixed_layer1_bias_v_read_readvariableop9
5savev2_adam_fixed_layer2_kernel_v_read_readvariableop7
3savev2_adam_fixed_layer2_bias_v_read_readvariableop:
6savev2_adam_action_output_kernel_v_read_readvariableop8
4savev2_adam_action_output_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
: *
dtype0*S
valueJBH B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_price_layer1_kernel_read_readvariableop,savev2_price_layer1_bias_read_readvariableop.savev2_fixed_layer1_kernel_read_readvariableop,savev2_fixed_layer1_bias_read_readvariableop.savev2_fixed_layer2_kernel_read_readvariableop,savev2_fixed_layer2_bias_read_readvariableop/savev2_action_output_kernel_read_readvariableop-savev2_action_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_price_layer1_kernel_m_read_readvariableop3savev2_adam_price_layer1_bias_m_read_readvariableop5savev2_adam_fixed_layer1_kernel_m_read_readvariableop3savev2_adam_fixed_layer1_bias_m_read_readvariableop5savev2_adam_fixed_layer2_kernel_m_read_readvariableop3savev2_adam_fixed_layer2_bias_m_read_readvariableop6savev2_adam_action_output_kernel_m_read_readvariableop4savev2_adam_action_output_bias_m_read_readvariableop5savev2_adam_price_layer1_kernel_v_read_readvariableop3savev2_adam_price_layer1_bias_v_read_readvariableop5savev2_adam_fixed_layer1_kernel_v_read_readvariableop3savev2_adam_fixed_layer1_bias_v_read_readvariableop5savev2_adam_fixed_layer2_kernel_v_read_readvariableop3savev2_adam_fixed_layer2_bias_v_read_readvariableop6savev2_adam_action_output_kernel_v_read_readvariableop4savev2_adam_action_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *.
dtypes$
"2 	2
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
_input_shapes?
?: : : :":::::: : : : : : : : : :":::::: : :":::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:": 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :
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
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:": 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:": 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
: 
?	
?
+__inference_model_3_layer_call_fn_475701630
price_input
	env_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_3_layer_call_and_return_conditional_losses_4757016112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_475701487

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
?
+__inference_model_3_layer_call_fn_475701581
price_input
	env_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_3_layer_call_and_return_conditional_losses_4757015622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?	
?
+__inference_model_3_layer_call_fn_475701756
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_3_layer_call_and_return_conditional_losses_4757015622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?/
?
F__inference_model_3_layer_call_and_return_conditional_losses_475701698
inputs_0
inputs_1/
+price_layer1_matmul_readvariableop_resource0
,price_layer1_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?#price_layer1/BiasAdd/ReadVariableOp?"price_layer1/MatMul/ReadVariableOp?
"price_layer1/MatMul/ReadVariableOpReadVariableOp+price_layer1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"price_layer1/MatMul/ReadVariableOp?
price_layer1/MatMulMatMulinputs_0*price_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer1/MatMul?
#price_layer1/BiasAdd/ReadVariableOpReadVariableOp,price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#price_layer1/BiasAdd/ReadVariableOp?
price_layer1/BiasAddBiasAddprice_layer1/MatMul:product:0+price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer1/BiasAdd
price_layer1/ReluReluprice_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
price_layer1/Relu{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
price_flatten/Const?
price_flatten/ReshapeReshapeprice_layer1/Relu:activations:0price_flatten/Const:output:0*
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
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp$^price_layer1/BiasAdd/ReadVariableOp#^price_layer1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2J
#price_layer1/BiasAdd/ReadVariableOp#price_layer1/BiasAdd/ReadVariableOp2H
"price_layer1/MatMul/ReadVariableOp"price_layer1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?/
?
F__inference_model_3_layer_call_and_return_conditional_losses_475701734
inputs_0
inputs_1/
+price_layer1_matmul_readvariableop_resource0
,price_layer1_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?#price_layer1/BiasAdd/ReadVariableOp?"price_layer1/MatMul/ReadVariableOp?
"price_layer1/MatMul/ReadVariableOpReadVariableOp+price_layer1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02$
"price_layer1/MatMul/ReadVariableOp?
price_layer1/MatMulMatMulinputs_0*price_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer1/MatMul?
#price_layer1/BiasAdd/ReadVariableOpReadVariableOp,price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#price_layer1/BiasAdd/ReadVariableOp?
price_layer1/BiasAddBiasAddprice_layer1/MatMul:product:0+price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer1/BiasAdd
price_layer1/ReluReluprice_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
price_layer1/Relu{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
price_flatten/Const?
price_flatten/ReshapeReshapeprice_layer1/Relu:activations:0price_flatten/Const:output:0*
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
action_output/BiasAdd?
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp$^price_layer1/BiasAdd/ReadVariableOp#^price_layer1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2J
#price_layer1/BiasAdd/ReadVariableOp#price_layer1/BiasAdd/ReadVariableOp2H
"price_layer1/MatMul/ReadVariableOp"price_layer1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:?????????
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_475701434

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
?
?
0__inference_price_layer1_layer_call_fn_475701798

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
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_4757013772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
+__inference_model_3_layer_call_fn_475701778
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_model_3_layer_call_and_return_conditional_losses_4757016112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_475701833

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
?
?
0__inference_fixed_layer2_layer_call_fn_475701862

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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_4757014612
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
?6
?
$__inference__wrapped_model_475701361
price_input
	env_input7
3model_3_price_layer1_matmul_readvariableop_resource8
4model_3_price_layer1_biasadd_readvariableop_resource7
3model_3_fixed_layer1_matmul_readvariableop_resource8
4model_3_fixed_layer1_biasadd_readvariableop_resource7
3model_3_fixed_layer2_matmul_readvariableop_resource8
4model_3_fixed_layer2_biasadd_readvariableop_resource8
4model_3_action_output_matmul_readvariableop_resource9
5model_3_action_output_biasadd_readvariableop_resource
identity??,model_3/action_output/BiasAdd/ReadVariableOp?+model_3/action_output/MatMul/ReadVariableOp?+model_3/fixed_layer1/BiasAdd/ReadVariableOp?*model_3/fixed_layer1/MatMul/ReadVariableOp?+model_3/fixed_layer2/BiasAdd/ReadVariableOp?*model_3/fixed_layer2/MatMul/ReadVariableOp?+model_3/price_layer1/BiasAdd/ReadVariableOp?*model_3/price_layer1/MatMul/ReadVariableOp?
*model_3/price_layer1/MatMul/ReadVariableOpReadVariableOp3model_3_price_layer1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*model_3/price_layer1/MatMul/ReadVariableOp?
model_3/price_layer1/MatMulMatMulprice_input2model_3/price_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_3/price_layer1/MatMul?
+model_3/price_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_3_price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_3/price_layer1/BiasAdd/ReadVariableOp?
model_3/price_layer1/BiasAddBiasAdd%model_3/price_layer1/MatMul:product:03model_3/price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_3/price_layer1/BiasAdd?
model_3/price_layer1/ReluRelu%model_3/price_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_3/price_layer1/Relu?
model_3/price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model_3/price_flatten/Const?
model_3/price_flatten/ReshapeReshape'model_3/price_layer1/Relu:activations:0$model_3/price_flatten/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_3/price_flatten/Reshape?
 model_3/concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_3/concat_layer/concat/axis?
model_3/concat_layer/concatConcatV2&model_3/price_flatten/Reshape:output:0	env_input)model_3/concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????"2
model_3/concat_layer/concat?
*model_3/fixed_layer1/MatMul/ReadVariableOpReadVariableOp3model_3_fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:"*
dtype02,
*model_3/fixed_layer1/MatMul/ReadVariableOp?
model_3/fixed_layer1/MatMulMatMul$model_3/concat_layer/concat:output:02model_3/fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/fixed_layer1/MatMul?
+model_3/fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_3_fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_3/fixed_layer1/BiasAdd/ReadVariableOp?
model_3/fixed_layer1/BiasAddBiasAdd%model_3/fixed_layer1/MatMul:product:03model_3/fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/fixed_layer1/BiasAdd?
model_3/fixed_layer1/ReluRelu%model_3/fixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/fixed_layer1/Relu?
*model_3/fixed_layer2/MatMul/ReadVariableOpReadVariableOp3model_3_fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*model_3/fixed_layer2/MatMul/ReadVariableOp?
model_3/fixed_layer2/MatMulMatMul'model_3/fixed_layer1/Relu:activations:02model_3/fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/fixed_layer2/MatMul?
+model_3/fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_3_fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_3/fixed_layer2/BiasAdd/ReadVariableOp?
model_3/fixed_layer2/BiasAddBiasAdd%model_3/fixed_layer2/MatMul:product:03model_3/fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/fixed_layer2/BiasAdd?
model_3/fixed_layer2/ReluRelu%model_3/fixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_3/fixed_layer2/Relu?
+model_3/action_output/MatMul/ReadVariableOpReadVariableOp4model_3_action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+model_3/action_output/MatMul/ReadVariableOp?
model_3/action_output/MatMulMatMul'model_3/fixed_layer2/Relu:activations:03model_3/action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/action_output/MatMul?
,model_3/action_output/BiasAdd/ReadVariableOpReadVariableOp5model_3_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_3/action_output/BiasAdd/ReadVariableOp?
model_3/action_output/BiasAddBiasAdd&model_3/action_output/MatMul:product:04model_3/action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_3/action_output/BiasAdd?
IdentityIdentity&model_3/action_output/BiasAdd:output:0-^model_3/action_output/BiasAdd/ReadVariableOp,^model_3/action_output/MatMul/ReadVariableOp,^model_3/fixed_layer1/BiasAdd/ReadVariableOp+^model_3/fixed_layer1/MatMul/ReadVariableOp,^model_3/fixed_layer2/BiasAdd/ReadVariableOp+^model_3/fixed_layer2/MatMul/ReadVariableOp,^model_3/price_layer1/BiasAdd/ReadVariableOp+^model_3/price_layer1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2\
,model_3/action_output/BiasAdd/ReadVariableOp,model_3/action_output/BiasAdd/ReadVariableOp2Z
+model_3/action_output/MatMul/ReadVariableOp+model_3/action_output/MatMul/ReadVariableOp2Z
+model_3/fixed_layer1/BiasAdd/ReadVariableOp+model_3/fixed_layer1/BiasAdd/ReadVariableOp2X
*model_3/fixed_layer1/MatMul/ReadVariableOp*model_3/fixed_layer1/MatMul/ReadVariableOp2Z
+model_3/fixed_layer2/BiasAdd/ReadVariableOp+model_3/fixed_layer2/BiasAdd/ReadVariableOp2X
*model_3/fixed_layer2/MatMul/ReadVariableOp*model_3/fixed_layer2/MatMul/ReadVariableOp2Z
+model_3/price_layer1/BiasAdd/ReadVariableOp+model_3/price_layer1/BiasAdd/ReadVariableOp2X
*model_3/price_layer1/MatMul/ReadVariableOp*model_3/price_layer1/MatMul/ReadVariableOp:T P
'
_output_shapes
:?????????
%
_user_specified_nameprice_input:RN
'
_output_shapes
:?????????
#
_user_specified_name	env_input
?
?
F__inference_model_3_layer_call_and_return_conditional_losses_475701611

inputs
inputs_1
price_layer1_475701588
price_layer1_475701590
fixed_layer1_475701595
fixed_layer1_475701597
fixed_layer2_475701600
fixed_layer2_475701602
action_output_475701605
action_output_475701607
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_475701588price_layer1_475701590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_4757013772&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_4757013992
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_4757014142
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_475701595fixed_layer1_475701597*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_4757014342&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_475701600fixed_layer2_475701602*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_4757014612&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_475701605action_output_475701607*
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
L__inference_action_output_layer_call_and_return_conditional_losses_4757014872'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_price_flatten_layer_call_fn_475701809

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
L__inference_price_flatten_layer_call_and_return_conditional_losses_4757013992
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
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_475701853

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
?	
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_475701789

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_475701461

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
?	
?
K__inference_price_layer1_layer_call_and_return_conditional_losses_475701377

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
L__inference_action_output_layer_call_and_return_conditional_losses_475701872

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
?
h
L__inference_price_flatten_layer_call_and_return_conditional_losses_475701804

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
'__inference_signature_wrapper_475701662
	env_input
price_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallprice_input	env_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__wrapped_model_4757013612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	env_input:TP
'
_output_shapes
:?????????
%
_user_specified_nameprice_input
?
\
0__inference_concat_layer_layer_call_fn_475701822
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_4757014142
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
?
?
F__inference_model_3_layer_call_and_return_conditional_losses_475701562

inputs
inputs_1
price_layer1_475701539
price_layer1_475701541
fixed_layer1_475701546
fixed_layer1_475701548
fixed_layer2_475701551
fixed_layer2_475701553
action_output_475701556
action_output_475701558
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_475701539price_layer1_475701541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_price_layer1_layer_call_and_return_conditional_losses_4757013772&
$price_layer1/StatefulPartitionedCall?
price_flatten/PartitionedCallPartitionedCall-price_layer1/StatefulPartitionedCall:output:0*
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_4757013992
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_4757014142
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_475701546fixed_layer1_475701548*
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_4757014342&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_475701551fixed_layer2_475701553*
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_4757014612&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_475701556action_output_475701558*
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
L__inference_action_output_layer_call_and_return_conditional_losses_4757014872'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Y
_input_shapesH
F:?????????:?????????::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
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
C
price_input4
serving_default_price_input:0?????????A
action_output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?;
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*m&call_and_return_all_conditional_losses
n_default_save_signature
o__call__"?8
_tf_keras_network?8{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}}, "training_config": {"loss": {"action_output": "mse"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "price_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*p&call_and_return_all_conditional_losses
q__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "price_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
*r&call_and_return_all_conditional_losses
s__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "price_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "env_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}}
?
	variables
trainable_variables
regularization_losses
	keras_api
*t&call_and_return_all_conditional_losses
u__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 2]}]}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*v&call_and_return_all_conditional_losses
w__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34]}}
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*x&call_and_return_all_conditional_losses
y__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*z&call_and_return_all_conditional_losses
{__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "action_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?
0iter

1beta_1

2beta_2
	3decay
4learning_ratem]m^m_m`$ma%mb*mc+mdvevfvgvh$vi%vj*vk+vl"
	optimizer
 "
trackable_dict_wrapper
X
0
1
2
3
$4
%5
*6
+7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
*6
+7"
trackable_list_wrapper
 "
trackable_list_wrapper
?

5layers
	variables
6layer_metrics
7layer_regularization_losses
trainable_variables
8non_trainable_variables
9metrics
regularization_losses
o__call__
n_default_save_signature
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
,
|serving_default"
signature_map
%:# 2price_layer1/kernel
: 2price_layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

:layers
	variables
;layer_metrics
<layer_regularization_losses
trainable_variables
=non_trainable_variables
>metrics
regularization_losses
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?layers
	variables
@layer_metrics
Alayer_regularization_losses
trainable_variables
Bnon_trainable_variables
Cmetrics
regularization_losses
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Dlayers
	variables
Elayer_metrics
Flayer_regularization_losses
trainable_variables
Gnon_trainable_variables
Hmetrics
regularization_losses
u__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
%:#"2fixed_layer1/kernel
:2fixed_layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ilayers
 	variables
Jlayer_metrics
Klayer_regularization_losses
!trainable_variables
Lnon_trainable_variables
Mmetrics
"regularization_losses
w__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
%:#2fixed_layer2/kernel
:2fixed_layer2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Nlayers
&	variables
Olayer_metrics
Player_regularization_losses
'trainable_variables
Qnon_trainable_variables
Rmetrics
(regularization_losses
y__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
&:$2action_output/kernel
 :2action_output/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Slayers
,	variables
Tlayer_metrics
Ulayer_regularization_losses
-trainable_variables
Vnon_trainable_variables
Wmetrics
.regularization_losses
{__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
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
?
	Ytotal
	Zcount
[	variables
\	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
Y0
Z1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
*:( 2Adam/price_layer1/kernel/m
$:" 2Adam/price_layer1/bias/m
*:("2Adam/fixed_layer1/kernel/m
$:"2Adam/fixed_layer1/bias/m
*:(2Adam/fixed_layer2/kernel/m
$:"2Adam/fixed_layer2/bias/m
+:)2Adam/action_output/kernel/m
%:#2Adam/action_output/bias/m
*:( 2Adam/price_layer1/kernel/v
$:" 2Adam/price_layer1/bias/v
*:("2Adam/fixed_layer1/kernel/v
$:"2Adam/fixed_layer1/bias/v
*:(2Adam/fixed_layer2/kernel/v
$:"2Adam/fixed_layer2/bias/v
+:)2Adam/action_output/kernel/v
%:#2Adam/action_output/bias/v
?2?
F__inference_model_3_layer_call_and_return_conditional_losses_475701504
F__inference_model_3_layer_call_and_return_conditional_losses_475701698
F__inference_model_3_layer_call_and_return_conditional_losses_475701734
F__inference_model_3_layer_call_and_return_conditional_losses_475701531?
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
$__inference__wrapped_model_475701361?
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
annotations? *T?Q
O?L
%?"
price_input?????????
#? 
	env_input?????????
?2?
+__inference_model_3_layer_call_fn_475701581
+__inference_model_3_layer_call_fn_475701630
+__inference_model_3_layer_call_fn_475701778
+__inference_model_3_layer_call_fn_475701756?
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
?2?
K__inference_price_layer1_layer_call_and_return_conditional_losses_475701789?
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
0__inference_price_layer1_layer_call_fn_475701798?
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
L__inference_price_flatten_layer_call_and_return_conditional_losses_475701804?
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
1__inference_price_flatten_layer_call_fn_475701809?
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
K__inference_concat_layer_layer_call_and_return_conditional_losses_475701816?
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
0__inference_concat_layer_layer_call_fn_475701822?
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
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_475701833?
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
0__inference_fixed_layer1_layer_call_fn_475701842?
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
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_475701853?
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
0__inference_fixed_layer2_layer_call_fn_475701862?
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
L__inference_action_output_layer_call_and_return_conditional_losses_475701872?
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
1__inference_action_output_layer_call_fn_475701881?
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
'__inference_signature_wrapper_475701662	env_inputprice_input"?
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
$__inference__wrapped_model_475701361?$%*+^?[
T?Q
O?L
%?"
price_input?????????
#? 
	env_input?????????
? "=?:
8
action_output'?$
action_output??????????
L__inference_action_output_layer_call_and_return_conditional_losses_475701872\*+/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_action_output_layer_call_fn_475701881O*+/?,
%?"
 ?
inputs?????????
? "???????????
K__inference_concat_layer_layer_call_and_return_conditional_losses_475701816?Z?W
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
0__inference_concat_layer_layer_call_fn_475701822vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????"?
K__inference_fixed_layer1_layer_call_and_return_conditional_losses_475701833\/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????
? ?
0__inference_fixed_layer1_layer_call_fn_475701842O/?,
%?"
 ?
inputs?????????"
? "???????????
K__inference_fixed_layer2_layer_call_and_return_conditional_losses_475701853\$%/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
0__inference_fixed_layer2_layer_call_fn_475701862O$%/?,
%?"
 ?
inputs?????????
? "???????????
F__inference_model_3_layer_call_and_return_conditional_losses_475701504?$%*+f?c
\?Y
O?L
%?"
price_input?????????
#? 
	env_input?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_model_3_layer_call_and_return_conditional_losses_475701531?$%*+f?c
\?Y
O?L
%?"
price_input?????????
#? 
	env_input?????????
p 

 
? "%?"
?
0?????????
? ?
F__inference_model_3_layer_call_and_return_conditional_losses_475701698?$%*+b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
F__inference_model_3_layer_call_and_return_conditional_losses_475701734?$%*+b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
+__inference_model_3_layer_call_fn_475701581?$%*+f?c
\?Y
O?L
%?"
price_input?????????
#? 
	env_input?????????
p

 
? "???????????
+__inference_model_3_layer_call_fn_475701630?$%*+f?c
\?Y
O?L
%?"
price_input?????????
#? 
	env_input?????????
p 

 
? "???????????
+__inference_model_3_layer_call_fn_475701756?$%*+b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
+__inference_model_3_layer_call_fn_475701778?$%*+b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
L__inference_price_flatten_layer_call_and_return_conditional_losses_475701804X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
1__inference_price_flatten_layer_call_fn_475701809K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
K__inference_price_layer1_layer_call_and_return_conditional_losses_475701789\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? ?
0__inference_price_layer1_layer_call_fn_475701798O/?,
%?"
 ?
inputs?????????
? "?????????? ?
'__inference_signature_wrapper_475701662?$%*+u?r
? 
k?h
0
	env_input#? 
	env_input?????????
4
price_input%?"
price_input?????????"=?:
8
action_output'?$
action_output?????????