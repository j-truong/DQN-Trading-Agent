¾	
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
 ?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??
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
price_layer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *$
shared_nameprice_layer2/kernel
{
'price_layer2/kernel/Read/ReadVariableOpReadVariableOpprice_layer2/kernel*
_output_shapes

:  *
dtype0
z
price_layer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameprice_layer2/bias
s
%price_layer2/bias/Read/ReadVariableOpReadVariableOpprice_layer2/bias*
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
Adam/price_layer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_nameAdam/price_layer2/kernel/m
?
.Adam/price_layer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/price_layer2/kernel/m*
_output_shapes

:  *
dtype0
?
Adam/price_layer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/price_layer2/bias/m
?
,Adam/price_layer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/price_layer2/bias/m*
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
Adam/price_layer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *+
shared_nameAdam/price_layer2/kernel/v
?
.Adam/price_layer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/price_layer2/kernel/v*
_output_shapes

:  *
dtype0
?
Adam/price_layer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/price_layer2/bias/v
?
,Adam/price_layer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/price_layer2/bias/v*
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
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
 	keras_api
 
R
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
h

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemimjmkml%mm&mn+mo,mp1mq2mrvsvtvuvv%vw&vx+vy,vz1v{2v|
 
F
0
1
2
3
%4
&5
+6
,7
18
29
F
0
1
2
3
%4
&5
+6
,7
18
29
 
?

<layers
	variables
=layer_metrics
>layer_regularization_losses
trainable_variables
?non_trainable_variables
@metrics
regularization_losses
 
_]
VARIABLE_VALUEprice_layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

Alayers
	variables
Blayer_metrics
Clayer_regularization_losses
trainable_variables
Dnon_trainable_variables
Emetrics
regularization_losses
_]
VARIABLE_VALUEprice_layer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEprice_layer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?

Flayers
	variables
Glayer_metrics
Hlayer_regularization_losses
trainable_variables
Inon_trainable_variables
Jmetrics
regularization_losses
 
 
 
?

Klayers
	variables
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
Nnon_trainable_variables
Ometrics
regularization_losses
 
 
 
?

Players
!	variables
Qlayer_metrics
Rlayer_regularization_losses
"trainable_variables
Snon_trainable_variables
Tmetrics
#regularization_losses
_]
VARIABLE_VALUEfixed_layer1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1

%0
&1
 
?

Ulayers
'	variables
Vlayer_metrics
Wlayer_regularization_losses
(trainable_variables
Xnon_trainable_variables
Ymetrics
)regularization_losses
_]
VARIABLE_VALUEfixed_layer2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEfixed_layer2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
?

Zlayers
-	variables
[layer_metrics
\layer_regularization_losses
.trainable_variables
]non_trainable_variables
^metrics
/regularization_losses
`^
VARIABLE_VALUEaction_output/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEaction_output/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

10
21

10
21
 
?

_layers
3	variables
`layer_metrics
alayer_regularization_losses
4trainable_variables
bnon_trainable_variables
cmetrics
5regularization_losses
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

d0
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
	etotal
	fcount
g	variables
h	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

e0
f1

g	variables
??
VARIABLE_VALUEAdam/price_layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/price_layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/price_layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/price_layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/action_output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/action_output/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/price_layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/price_layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/price_layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/price_layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/fixed_layer2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/fixed_layer2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/action_output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/action_output/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
GPU 2J 8? *1
f,R*
(__inference_signature_wrapper_1195680872
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'price_layer1/kernel/Read/ReadVariableOp%price_layer1/bias/Read/ReadVariableOp'price_layer2/kernel/Read/ReadVariableOp%price_layer2/bias/Read/ReadVariableOp'fixed_layer1/kernel/Read/ReadVariableOp%fixed_layer1/bias/Read/ReadVariableOp'fixed_layer2/kernel/Read/ReadVariableOp%fixed_layer2/bias/Read/ReadVariableOp(action_output/kernel/Read/ReadVariableOp&action_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/price_layer1/kernel/m/Read/ReadVariableOp,Adam/price_layer1/bias/m/Read/ReadVariableOp.Adam/price_layer2/kernel/m/Read/ReadVariableOp,Adam/price_layer2/bias/m/Read/ReadVariableOp.Adam/fixed_layer1/kernel/m/Read/ReadVariableOp,Adam/fixed_layer1/bias/m/Read/ReadVariableOp.Adam/fixed_layer2/kernel/m/Read/ReadVariableOp,Adam/fixed_layer2/bias/m/Read/ReadVariableOp/Adam/action_output/kernel/m/Read/ReadVariableOp-Adam/action_output/bias/m/Read/ReadVariableOp.Adam/price_layer1/kernel/v/Read/ReadVariableOp,Adam/price_layer1/bias/v/Read/ReadVariableOp.Adam/price_layer2/kernel/v/Read/ReadVariableOp,Adam/price_layer2/bias/v/Read/ReadVariableOp.Adam/fixed_layer1/kernel/v/Read/ReadVariableOp,Adam/fixed_layer1/bias/v/Read/ReadVariableOp.Adam/fixed_layer2/kernel/v/Read/ReadVariableOp,Adam/fixed_layer2/bias/v/Read/ReadVariableOp/Adam/action_output/kernel/v/Read/ReadVariableOp-Adam/action_output/bias/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
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
GPU 2J 8? *,
f'R%
#__inference__traced_save_1195681268
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameprice_layer1/kernelprice_layer1/biasprice_layer2/kernelprice_layer2/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/price_layer1/kernel/mAdam/price_layer1/bias/mAdam/price_layer2/kernel/mAdam/price_layer2/bias/mAdam/fixed_layer1/kernel/mAdam/fixed_layer1/bias/mAdam/fixed_layer2/kernel/mAdam/fixed_layer2/bias/mAdam/action_output/kernel/mAdam/action_output/bias/mAdam/price_layer1/kernel/vAdam/price_layer1/bias/vAdam/price_layer2/kernel/vAdam/price_layer2/bias/vAdam/fixed_layer1/kernel/vAdam/fixed_layer1/bias/vAdam/fixed_layer2/kernel/vAdam/fixed_layer2/bias/vAdam/action_output/kernel/vAdam/action_output/bias/v*1
Tin*
(2&*
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
GPU 2J 8? */
f*R(
&__inference__traced_restore_1195681389??
?
?
2__inference_action_output_layer_call_fn_1195681133

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
GPU 2J 8? *V
fQRO
M__inference_action_output_layer_call_and_return_conditional_losses_11956806702
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
?9
?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680958
inputs_0
inputs_1/
+price_layer1_matmul_readvariableop_resource0
,price_layer1_biasadd_readvariableop_resource/
+price_layer2_matmul_readvariableop_resource0
,price_layer2_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?#price_layer1/BiasAdd/ReadVariableOp?"price_layer1/MatMul/ReadVariableOp?#price_layer2/BiasAdd/ReadVariableOp?"price_layer2/MatMul/ReadVariableOp?
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
price_layer1/Relu?
"price_layer2/MatMul/ReadVariableOpReadVariableOp+price_layer2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"price_layer2/MatMul/ReadVariableOp?
price_layer2/MatMulMatMulprice_layer1/Relu:activations:0*price_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer2/MatMul?
#price_layer2/BiasAdd/ReadVariableOpReadVariableOp,price_layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#price_layer2/BiasAdd/ReadVariableOp?
price_layer2/BiasAddBiasAddprice_layer2/MatMul:product:0+price_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer2/BiasAdd
price_layer2/ReluReluprice_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/Relu{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
price_flatten/Const?
price_flatten/ReshapeReshapeprice_layer2/Relu:activations:0price_flatten/Const:output:0*
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
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp$^price_layer1/BiasAdd/ReadVariableOp#^price_layer1/MatMul/ReadVariableOp$^price_layer2/BiasAdd/ReadVariableOp#^price_layer2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2J
#price_layer1/BiasAdd/ReadVariableOp#price_layer1/BiasAdd/ReadVariableOp2H
"price_layer1/MatMul/ReadVariableOp"price_layer1/MatMul/ReadVariableOp2J
#price_layer2/BiasAdd/ReadVariableOp#price_layer2/BiasAdd/ReadVariableOp2H
"price_layer2/MatMul/ReadVariableOp"price_layer2/MatMul/ReadVariableOp:Q M
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
??
?
&__inference__traced_restore_1195681389
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
%assignvariableop_9_action_output_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count2
.assignvariableop_17_adam_price_layer1_kernel_m0
,assignvariableop_18_adam_price_layer1_bias_m2
.assignvariableop_19_adam_price_layer2_kernel_m0
,assignvariableop_20_adam_price_layer2_bias_m2
.assignvariableop_21_adam_fixed_layer1_kernel_m0
,assignvariableop_22_adam_fixed_layer1_bias_m2
.assignvariableop_23_adam_fixed_layer2_kernel_m0
,assignvariableop_24_adam_fixed_layer2_bias_m3
/assignvariableop_25_adam_action_output_kernel_m1
-assignvariableop_26_adam_action_output_bias_m2
.assignvariableop_27_adam_price_layer1_kernel_v0
,assignvariableop_28_adam_price_layer1_bias_v2
.assignvariableop_29_adam_price_layer2_kernel_v0
,assignvariableop_30_adam_price_layer2_bias_v2
.assignvariableop_31_adam_fixed_layer1_kernel_v0
,assignvariableop_32_adam_fixed_layer1_bias_v2
.assignvariableop_33_adam_fixed_layer2_kernel_v0
,assignvariableop_34_adam_fixed_layer2_bias_v3
/assignvariableop_35_adam_action_output_kernel_v1
-assignvariableop_36_adam_action_output_bias_v
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
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
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_adam_price_layer1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_adam_price_layer1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_price_layer2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_price_layer2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_fixed_layer1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_fixed_layer1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp.assignvariableop_23_adam_fixed_layer2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_adam_fixed_layer2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp/assignvariableop_25_adam_action_output_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp-assignvariableop_26_adam_action_output_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_price_layer1_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp,assignvariableop_28_adam_price_layer1_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp.assignvariableop_29_adam_price_layer2_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_price_layer2_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp.assignvariableop_31_adam_fixed_layer1_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp,assignvariableop_32_adam_fixed_layer1_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp.assignvariableop_33_adam_fixed_layer2_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp,assignvariableop_34_adam_fixed_layer2_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_adam_action_output_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp-assignvariableop_36_adam_action_output_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
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
L__inference_price_layer1_layer_call_and_return_conditional_losses_1195681021

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
?
N
2__inference_price_flatten_layer_call_fn_1195681061

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
GPU 2J 8? *V
fQRO
M__inference_price_flatten_layer_call_and_return_conditional_losses_11956805822
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
?%
?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680687
price_input
	env_input
price_layer1_1195680544
price_layer1_1195680546
price_layer2_1195680571
price_layer2_1195680573
fixed_layer1_1195680628
fixed_layer1_1195680630
fixed_layer2_1195680655
fixed_layer2_1195680657
action_output_1195680681
action_output_1195680683
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_1195680544price_layer1_1195680546*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11956805332&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1195680571price_layer2_1195680573*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11956805602&
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
GPU 2J 8? *V
fQRO
M__inference_price_flatten_layer_call_and_return_conditional_losses_11956805822
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
GPU 2J 8? *U
fPRN
L__inference_concat_layer_layer_call_and_return_conditional_losses_11956805972
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1195680628fixed_layer1_1195680630*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11956806172&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1195680655fixed_layer2_1195680657*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11956806442&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1195680681action_output_1195680683*
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
GPU 2J 8? *V
fQRO
M__inference_action_output_layer_call_and_return_conditional_losses_11956806702'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:T P
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
1__inference_price_layer1_layer_call_fn_1195681030

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
GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11956805332
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1195680644

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
?
,__inference_model_9_layer_call_fn_1195680984
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11956807552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::22
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
?%
?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680813

inputs
inputs_1
price_layer1_1195680785
price_layer1_1195680787
price_layer2_1195680790
price_layer2_1195680792
fixed_layer1_1195680797
fixed_layer1_1195680799
fixed_layer2_1195680802
fixed_layer2_1195680804
action_output_1195680807
action_output_1195680809
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_1195680785price_layer1_1195680787*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11956805332&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1195680790price_layer2_1195680792*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11956805602&
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
GPU 2J 8? *V
fQRO
M__inference_price_flatten_layer_call_and_return_conditional_losses_11956805822
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
GPU 2J 8? *U
fPRN
L__inference_concat_layer_layer_call_and_return_conditional_losses_11956805972
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1195680797fixed_layer1_1195680799*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11956806172&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1195680802fixed_layer2_1195680804*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11956806442&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1195680807action_output_1195680809*
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
GPU 2J 8? *V
fQRO
M__inference_action_output_layer_call_and_return_conditional_losses_11956806702'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?Q
?
#__inference__traced_save_1195681268
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
5savev2_adam_price_layer2_kernel_m_read_readvariableop7
3savev2_adam_price_layer2_bias_m_read_readvariableop9
5savev2_adam_fixed_layer1_kernel_m_read_readvariableop7
3savev2_adam_fixed_layer1_bias_m_read_readvariableop9
5savev2_adam_fixed_layer2_kernel_m_read_readvariableop7
3savev2_adam_fixed_layer2_bias_m_read_readvariableop:
6savev2_adam_action_output_kernel_m_read_readvariableop8
4savev2_adam_action_output_bias_m_read_readvariableop9
5savev2_adam_price_layer1_kernel_v_read_readvariableop7
3savev2_adam_price_layer1_bias_v_read_readvariableop9
5savev2_adam_price_layer2_kernel_v_read_readvariableop7
3savev2_adam_price_layer2_bias_v_read_readvariableop9
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_price_layer1_kernel_read_readvariableop,savev2_price_layer1_bias_read_readvariableop.savev2_price_layer2_kernel_read_readvariableop,savev2_price_layer2_bias_read_readvariableop.savev2_fixed_layer1_kernel_read_readvariableop,savev2_fixed_layer1_bias_read_readvariableop.savev2_fixed_layer2_kernel_read_readvariableop,savev2_fixed_layer2_bias_read_readvariableop/savev2_action_output_kernel_read_readvariableop-savev2_action_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_price_layer1_kernel_m_read_readvariableop3savev2_adam_price_layer1_bias_m_read_readvariableop5savev2_adam_price_layer2_kernel_m_read_readvariableop3savev2_adam_price_layer2_bias_m_read_readvariableop5savev2_adam_fixed_layer1_kernel_m_read_readvariableop3savev2_adam_fixed_layer1_bias_m_read_readvariableop5savev2_adam_fixed_layer2_kernel_m_read_readvariableop3savev2_adam_fixed_layer2_bias_m_read_readvariableop6savev2_adam_action_output_kernel_m_read_readvariableop4savev2_adam_action_output_bias_m_read_readvariableop5savev2_adam_price_layer1_kernel_v_read_readvariableop3savev2_adam_price_layer1_bias_v_read_readvariableop5savev2_adam_price_layer2_kernel_v_read_readvariableop3savev2_adam_price_layer2_bias_v_read_readvariableop5savev2_adam_fixed_layer1_kernel_v_read_readvariableop3savev2_adam_fixed_layer1_bias_v_read_readvariableop5savev2_adam_fixed_layer2_kernel_v_read_readvariableop3savev2_adam_fixed_layer2_bias_v_read_readvariableop6savev2_adam_action_output_kernel_v_read_readvariableop4savev2_adam_action_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :  : :":::::: : : : : : : : : :  : :":::::: : :  : :":::::: 2(
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

:  : 

_output_shapes
: :$ 

_output_shapes

:": 
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
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:": 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$  

_output_shapes

:": !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$$ 

_output_shapes

:: %

_output_shapes
::&

_output_shapes
: 
?
i
M__inference_price_flatten_layer_call_and_return_conditional_losses_1195681056

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
L__inference_price_layer2_layer_call_and_return_conditional_losses_1195680560

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
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
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?%
?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680755

inputs
inputs_1
price_layer1_1195680727
price_layer1_1195680729
price_layer2_1195680732
price_layer2_1195680734
fixed_layer1_1195680739
fixed_layer1_1195680741
fixed_layer2_1195680744
fixed_layer2_1195680746
action_output_1195680749
action_output_1195680751
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_1195680727price_layer1_1195680729*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11956805332&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1195680732price_layer2_1195680734*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11956805602&
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
GPU 2J 8? *V
fQRO
M__inference_price_flatten_layer_call_and_return_conditional_losses_11956805822
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
GPU 2J 8? *U
fPRN
L__inference_concat_layer_layer_call_and_return_conditional_losses_11956805972
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1195680739fixed_layer1_1195680741*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11956806172&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1195680744fixed_layer2_1195680746*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11956806442&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1195680749action_output_1195680751*
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
GPU 2J 8? *V
fQRO
M__inference_action_output_layer_call_and_return_conditional_losses_11956806702'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
M__inference_action_output_layer_call_and_return_conditional_losses_1195681124

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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1195681105

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
?
,__inference_model_9_layer_call_fn_1195680778
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11956807552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::22
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
?
]
1__inference_concat_layer_layer_call_fn_1195681074
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
GPU 2J 8? *U
fPRN
L__inference_concat_layer_layer_call_and_return_conditional_losses_11956805972
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
?
i
M__inference_price_flatten_layer_call_and_return_conditional_losses_1195680582

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
L__inference_price_layer1_layer_call_and_return_conditional_losses_1195680533

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
?9
?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680915
inputs_0
inputs_1/
+price_layer1_matmul_readvariableop_resource0
,price_layer1_biasadd_readvariableop_resource/
+price_layer2_matmul_readvariableop_resource0
,price_layer2_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?#price_layer1/BiasAdd/ReadVariableOp?"price_layer1/MatMul/ReadVariableOp?#price_layer2/BiasAdd/ReadVariableOp?"price_layer2/MatMul/ReadVariableOp?
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
price_layer1/Relu?
"price_layer2/MatMul/ReadVariableOpReadVariableOp+price_layer2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02$
"price_layer2/MatMul/ReadVariableOp?
price_layer2/MatMulMatMulprice_layer1/Relu:activations:0*price_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer2/MatMul?
#price_layer2/BiasAdd/ReadVariableOpReadVariableOp,price_layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02%
#price_layer2/BiasAdd/ReadVariableOp?
price_layer2/BiasAddBiasAddprice_layer2/MatMul:product:0+price_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
price_layer2/BiasAdd
price_layer2/ReluReluprice_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/Relu{
price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
price_flatten/Const?
price_flatten/ReshapeReshapeprice_layer2/Relu:activations:0price_flatten/Const:output:0*
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
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp$^price_layer1/BiasAdd/ReadVariableOp#^price_layer1/MatMul/ReadVariableOp$^price_layer2/BiasAdd/ReadVariableOp#^price_layer2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2L
$action_output/BiasAdd/ReadVariableOp$action_output/BiasAdd/ReadVariableOp2J
#action_output/MatMul/ReadVariableOp#action_output/MatMul/ReadVariableOp2J
#fixed_layer1/BiasAdd/ReadVariableOp#fixed_layer1/BiasAdd/ReadVariableOp2H
"fixed_layer1/MatMul/ReadVariableOp"fixed_layer1/MatMul/ReadVariableOp2J
#fixed_layer2/BiasAdd/ReadVariableOp#fixed_layer2/BiasAdd/ReadVariableOp2H
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2J
#price_layer1/BiasAdd/ReadVariableOp#price_layer1/BiasAdd/ReadVariableOp2H
"price_layer1/MatMul/ReadVariableOp"price_layer1/MatMul/ReadVariableOp2J
#price_layer2/BiasAdd/ReadVariableOp#price_layer2/BiasAdd/ReadVariableOp2H
"price_layer2/MatMul/ReadVariableOp"price_layer2/MatMul/ReadVariableOp:Q M
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
L__inference_price_layer2_layer_call_and_return_conditional_losses_1195681041

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
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
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
,__inference_model_9_layer_call_fn_1195681010
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11956808132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::22
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
?
x
L__inference_concat_layer_layer_call_and_return_conditional_losses_1195681068
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
?
(__inference_signature_wrapper_1195680872
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
GPU 2J 8? *.
f)R'
%__inference__wrapped_model_11956805172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::22
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
?A
?
%__inference__wrapped_model_1195680517
price_input
	env_input7
3model_9_price_layer1_matmul_readvariableop_resource8
4model_9_price_layer1_biasadd_readvariableop_resource7
3model_9_price_layer2_matmul_readvariableop_resource8
4model_9_price_layer2_biasadd_readvariableop_resource7
3model_9_fixed_layer1_matmul_readvariableop_resource8
4model_9_fixed_layer1_biasadd_readvariableop_resource7
3model_9_fixed_layer2_matmul_readvariableop_resource8
4model_9_fixed_layer2_biasadd_readvariableop_resource8
4model_9_action_output_matmul_readvariableop_resource9
5model_9_action_output_biasadd_readvariableop_resource
identity??,model_9/action_output/BiasAdd/ReadVariableOp?+model_9/action_output/MatMul/ReadVariableOp?+model_9/fixed_layer1/BiasAdd/ReadVariableOp?*model_9/fixed_layer1/MatMul/ReadVariableOp?+model_9/fixed_layer2/BiasAdd/ReadVariableOp?*model_9/fixed_layer2/MatMul/ReadVariableOp?+model_9/price_layer1/BiasAdd/ReadVariableOp?*model_9/price_layer1/MatMul/ReadVariableOp?+model_9/price_layer2/BiasAdd/ReadVariableOp?*model_9/price_layer2/MatMul/ReadVariableOp?
*model_9/price_layer1/MatMul/ReadVariableOpReadVariableOp3model_9_price_layer1_matmul_readvariableop_resource*
_output_shapes

: *
dtype02,
*model_9/price_layer1/MatMul/ReadVariableOp?
model_9/price_layer1/MatMulMatMulprice_input2model_9/price_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer1/MatMul?
+model_9/price_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_9_price_layer1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_9/price_layer1/BiasAdd/ReadVariableOp?
model_9/price_layer1/BiasAddBiasAdd%model_9/price_layer1/MatMul:product:03model_9/price_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer1/BiasAdd?
model_9/price_layer1/ReluRelu%model_9/price_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer1/Relu?
*model_9/price_layer2/MatMul/ReadVariableOpReadVariableOp3model_9_price_layer2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02,
*model_9/price_layer2/MatMul/ReadVariableOp?
model_9/price_layer2/MatMulMatMul'model_9/price_layer1/Relu:activations:02model_9/price_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer2/MatMul?
+model_9/price_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_9_price_layer2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_9/price_layer2/BiasAdd/ReadVariableOp?
model_9/price_layer2/BiasAddBiasAdd%model_9/price_layer2/MatMul:product:03model_9/price_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer2/BiasAdd?
model_9/price_layer2/ReluRelu%model_9/price_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer2/Relu?
model_9/price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model_9/price_flatten/Const?
model_9/price_flatten/ReshapeReshape'model_9/price_layer2/Relu:activations:0$model_9/price_flatten/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_flatten/Reshape?
 model_9/concat_layer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_9/concat_layer/concat/axis?
model_9/concat_layer/concatConcatV2&model_9/price_flatten/Reshape:output:0	env_input)model_9/concat_layer/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????"2
model_9/concat_layer/concat?
*model_9/fixed_layer1/MatMul/ReadVariableOpReadVariableOp3model_9_fixed_layer1_matmul_readvariableop_resource*
_output_shapes

:"*
dtype02,
*model_9/fixed_layer1/MatMul/ReadVariableOp?
model_9/fixed_layer1/MatMulMatMul$model_9/concat_layer/concat:output:02model_9/fixed_layer1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_9/fixed_layer1/MatMul?
+model_9/fixed_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_9_fixed_layer1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_9/fixed_layer1/BiasAdd/ReadVariableOp?
model_9/fixed_layer1/BiasAddBiasAdd%model_9/fixed_layer1/MatMul:product:03model_9/fixed_layer1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_9/fixed_layer1/BiasAdd?
model_9/fixed_layer1/ReluRelu%model_9/fixed_layer1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_9/fixed_layer1/Relu?
*model_9/fixed_layer2/MatMul/ReadVariableOpReadVariableOp3model_9_fixed_layer2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*model_9/fixed_layer2/MatMul/ReadVariableOp?
model_9/fixed_layer2/MatMulMatMul'model_9/fixed_layer1/Relu:activations:02model_9/fixed_layer2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_9/fixed_layer2/MatMul?
+model_9/fixed_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_9_fixed_layer2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_9/fixed_layer2/BiasAdd/ReadVariableOp?
model_9/fixed_layer2/BiasAddBiasAdd%model_9/fixed_layer2/MatMul:product:03model_9/fixed_layer2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_9/fixed_layer2/BiasAdd?
model_9/fixed_layer2/ReluRelu%model_9/fixed_layer2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_9/fixed_layer2/Relu?
+model_9/action_output/MatMul/ReadVariableOpReadVariableOp4model_9_action_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+model_9/action_output/MatMul/ReadVariableOp?
model_9/action_output/MatMulMatMul'model_9/fixed_layer2/Relu:activations:03model_9/action_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_9/action_output/MatMul?
,model_9/action_output/BiasAdd/ReadVariableOpReadVariableOp5model_9_action_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_9/action_output/BiasAdd/ReadVariableOp?
model_9/action_output/BiasAddBiasAdd&model_9/action_output/MatMul:product:04model_9/action_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_9/action_output/BiasAdd?
IdentityIdentity&model_9/action_output/BiasAdd:output:0-^model_9/action_output/BiasAdd/ReadVariableOp,^model_9/action_output/MatMul/ReadVariableOp,^model_9/fixed_layer1/BiasAdd/ReadVariableOp+^model_9/fixed_layer1/MatMul/ReadVariableOp,^model_9/fixed_layer2/BiasAdd/ReadVariableOp+^model_9/fixed_layer2/MatMul/ReadVariableOp,^model_9/price_layer1/BiasAdd/ReadVariableOp+^model_9/price_layer1/MatMul/ReadVariableOp,^model_9/price_layer2/BiasAdd/ReadVariableOp+^model_9/price_layer2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2\
,model_9/action_output/BiasAdd/ReadVariableOp,model_9/action_output/BiasAdd/ReadVariableOp2Z
+model_9/action_output/MatMul/ReadVariableOp+model_9/action_output/MatMul/ReadVariableOp2Z
+model_9/fixed_layer1/BiasAdd/ReadVariableOp+model_9/fixed_layer1/BiasAdd/ReadVariableOp2X
*model_9/fixed_layer1/MatMul/ReadVariableOp*model_9/fixed_layer1/MatMul/ReadVariableOp2Z
+model_9/fixed_layer2/BiasAdd/ReadVariableOp+model_9/fixed_layer2/BiasAdd/ReadVariableOp2X
*model_9/fixed_layer2/MatMul/ReadVariableOp*model_9/fixed_layer2/MatMul/ReadVariableOp2Z
+model_9/price_layer1/BiasAdd/ReadVariableOp+model_9/price_layer1/BiasAdd/ReadVariableOp2X
*model_9/price_layer1/MatMul/ReadVariableOp*model_9/price_layer1/MatMul/ReadVariableOp2Z
+model_9/price_layer2/BiasAdd/ReadVariableOp+model_9/price_layer2/BiasAdd/ReadVariableOp2X
*model_9/price_layer2/MatMul/ReadVariableOp*model_9/price_layer2/MatMul/ReadVariableOp:T P
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
?%
?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680719
price_input
	env_input
price_layer1_1195680691
price_layer1_1195680693
price_layer2_1195680696
price_layer2_1195680698
fixed_layer1_1195680703
fixed_layer1_1195680705
fixed_layer2_1195680708
fixed_layer2_1195680710
action_output_1195680713
action_output_1195680715
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_1195680691price_layer1_1195680693*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11956805332&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1195680696price_layer2_1195680698*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11956805602&
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
GPU 2J 8? *V
fQRO
M__inference_price_flatten_layer_call_and_return_conditional_losses_11956805822
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
GPU 2J 8? *U
fPRN
L__inference_concat_layer_layer_call_and_return_conditional_losses_11956805972
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1195680703fixed_layer1_1195680705*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11956806172&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1195680708fixed_layer2_1195680710*
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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11956806442&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1195680713action_output_1195680715*
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
GPU 2J 8? *V
fQRO
M__inference_action_output_layer_call_and_return_conditional_losses_11956806702'
%action_output/StatefulPartitionedCall?
IdentityIdentity.action_output/StatefulPartitionedCall:output:0&^action_output/StatefulPartitionedCall%^fixed_layer1/StatefulPartitionedCall%^fixed_layer2/StatefulPartitionedCall%^price_layer1/StatefulPartitionedCall%^price_layer2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::2N
%action_output/StatefulPartitionedCall%action_output/StatefulPartitionedCall2L
$fixed_layer1/StatefulPartitionedCall$fixed_layer1/StatefulPartitionedCall2L
$fixed_layer2/StatefulPartitionedCall$fixed_layer2/StatefulPartitionedCall2L
$price_layer1/StatefulPartitionedCall$price_layer1/StatefulPartitionedCall2L
$price_layer2/StatefulPartitionedCall$price_layer2/StatefulPartitionedCall:T P
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
?
v
L__inference_concat_layer_layer_call_and_return_conditional_losses_1195680597

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
?	
?
,__inference_model_9_layer_call_fn_1195680836
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11956808132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*a
_input_shapesP
N:?????????:?????????::::::::::22
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
?
?
1__inference_fixed_layer1_layer_call_fn_1195681094

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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11956806172
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
?	
?
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1195681085

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
M__inference_action_output_layer_call_and_return_conditional_losses_1195680670

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
?
?
1__inference_fixed_layer2_layer_call_fn_1195681114

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
GPU 2J 8? *U
fPRN
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11956806442
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
?	
?
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1195680617

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
1__inference_price_layer2_layer_call_fn_1195681050

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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11956805602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
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
?D
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
*}&call_and_return_all_conditional_losses
~_default_save_signature
__call__"?@
_tf_keras_network?@{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}}, "training_config": {"loss": {"action_output": "mse"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "price_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "price_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "price_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
	variables
trainable_variables
regularization_losses
 	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "price_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "env_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}}
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concat_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 2]}]}
?

%kernel
&bias
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 34}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34]}}
?

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "fixed_layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
?

1kernel
2bias
3	variables
4trainable_variables
5regularization_losses
6	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "action_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
?
7iter

8beta_1

9beta_2
	:decay
;learning_ratemimjmkml%mm&mn+mo,mp1mq2mrvsvtvuvv%vw&vx+vy,vz1v{2v|"
	optimizer
 "
trackable_dict_wrapper
f
0
1
2
3
%4
&5
+6
,7
18
29"
trackable_list_wrapper
f
0
1
2
3
%4
&5
+6
,7
18
29"
trackable_list_wrapper
 "
trackable_list_wrapper
?

<layers
	variables
=layer_metrics
>layer_regularization_losses
trainable_variables
?non_trainable_variables
@metrics
regularization_losses
__call__
~_default_save_signature
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:# 2price_layer1/kernel
: 2price_layer1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Alayers
	variables
Blayer_metrics
Clayer_regularization_losses
trainable_variables
Dnon_trainable_variables
Emetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#  2price_layer2/kernel
: 2price_layer2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Flayers
	variables
Glayer_metrics
Hlayer_regularization_losses
trainable_variables
Inon_trainable_variables
Jmetrics
regularization_losses
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

Klayers
	variables
Llayer_metrics
Mlayer_regularization_losses
trainable_variables
Nnon_trainable_variables
Ometrics
regularization_losses
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

Players
!	variables
Qlayer_metrics
Rlayer_regularization_losses
"trainable_variables
Snon_trainable_variables
Tmetrics
#regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#"2fixed_layer1/kernel
:2fixed_layer1/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Ulayers
'	variables
Vlayer_metrics
Wlayer_regularization_losses
(trainable_variables
Xnon_trainable_variables
Ymetrics
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#2fixed_layer2/kernel
:2fixed_layer2/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
?

Zlayers
-	variables
[layer_metrics
\layer_regularization_losses
.trainable_variables
]non_trainable_variables
^metrics
/regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2action_output/kernel
 :2action_output/bias
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?

_layers
3	variables
`layer_metrics
alayer_regularization_losses
4trainable_variables
bnon_trainable_variables
cmetrics
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
d0"
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
?
	etotal
	fcount
g	variables
h	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
*:( 2Adam/price_layer1/kernel/m
$:" 2Adam/price_layer1/bias/m
*:(  2Adam/price_layer2/kernel/m
$:" 2Adam/price_layer2/bias/m
*:("2Adam/fixed_layer1/kernel/m
$:"2Adam/fixed_layer1/bias/m
*:(2Adam/fixed_layer2/kernel/m
$:"2Adam/fixed_layer2/bias/m
+:)2Adam/action_output/kernel/m
%:#2Adam/action_output/bias/m
*:( 2Adam/price_layer1/kernel/v
$:" 2Adam/price_layer1/bias/v
*:(  2Adam/price_layer2/kernel/v
$:" 2Adam/price_layer2/bias/v
*:("2Adam/fixed_layer1/kernel/v
$:"2Adam/fixed_layer1/bias/v
*:(2Adam/fixed_layer2/kernel/v
$:"2Adam/fixed_layer2/bias/v
+:)2Adam/action_output/kernel/v
%:#2Adam/action_output/bias/v
?2?
G__inference_model_9_layer_call_and_return_conditional_losses_1195680719
G__inference_model_9_layer_call_and_return_conditional_losses_1195680915
G__inference_model_9_layer_call_and_return_conditional_losses_1195680958
G__inference_model_9_layer_call_and_return_conditional_losses_1195680687?
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
%__inference__wrapped_model_1195680517?
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
,__inference_model_9_layer_call_fn_1195680778
,__inference_model_9_layer_call_fn_1195680984
,__inference_model_9_layer_call_fn_1195680836
,__inference_model_9_layer_call_fn_1195681010?
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
L__inference_price_layer1_layer_call_and_return_conditional_losses_1195681021?
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
1__inference_price_layer1_layer_call_fn_1195681030?
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
L__inference_price_layer2_layer_call_and_return_conditional_losses_1195681041?
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
1__inference_price_layer2_layer_call_fn_1195681050?
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_1195681056?
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
2__inference_price_flatten_layer_call_fn_1195681061?
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_1195681068?
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
1__inference_concat_layer_layer_call_fn_1195681074?
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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1195681085?
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
1__inference_fixed_layer1_layer_call_fn_1195681094?
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1195681105?
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
1__inference_fixed_layer2_layer_call_fn_1195681114?
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
M__inference_action_output_layer_call_and_return_conditional_losses_1195681124?
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
2__inference_action_output_layer_call_fn_1195681133?
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
(__inference_signature_wrapper_1195680872	env_inputprice_input"?
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
%__inference__wrapped_model_1195680517?
%&+,12^?[
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
M__inference_action_output_layer_call_and_return_conditional_losses_1195681124\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
2__inference_action_output_layer_call_fn_1195681133O12/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_concat_layer_layer_call_and_return_conditional_losses_1195681068?Z?W
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
1__inference_concat_layer_layer_call_fn_1195681074vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????"?
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1195681085\%&/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????
? ?
1__inference_fixed_layer1_layer_call_fn_1195681094O%&/?,
%?"
 ?
inputs?????????"
? "???????????
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1195681105\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_fixed_layer2_layer_call_fn_1195681114O+,/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_model_9_layer_call_and_return_conditional_losses_1195680687?
%&+,12f?c
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
G__inference_model_9_layer_call_and_return_conditional_losses_1195680719?
%&+,12f?c
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
G__inference_model_9_layer_call_and_return_conditional_losses_1195680915?
%&+,12b?_
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
G__inference_model_9_layer_call_and_return_conditional_losses_1195680958?
%&+,12b?_
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
,__inference_model_9_layer_call_fn_1195680778?
%&+,12f?c
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
,__inference_model_9_layer_call_fn_1195680836?
%&+,12f?c
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
,__inference_model_9_layer_call_fn_1195680984?
%&+,12b?_
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
,__inference_model_9_layer_call_fn_1195681010?
%&+,12b?_
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_1195681056X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
2__inference_price_flatten_layer_call_fn_1195681061K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1195681021\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? ?
1__inference_price_layer1_layer_call_fn_1195681030O/?,
%?"
 ?
inputs?????????
? "?????????? ?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1195681041\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
1__inference_price_layer2_layer_call_fn_1195681050O/?,
%?"
 ?
inputs????????? 
? "?????????? ?
(__inference_signature_wrapper_1195680872?
%&+,12u?r
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