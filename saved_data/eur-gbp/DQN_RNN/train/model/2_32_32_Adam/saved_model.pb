??*
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
?"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8??'
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
?
 price_layer1/lstm_cell_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*1
shared_name" price_layer1/lstm_cell_14/kernel
?
4price_layer1/lstm_cell_14/kernel/Read/ReadVariableOpReadVariableOp price_layer1/lstm_cell_14/kernel*
_output_shapes
:	?*
dtype0
?
*price_layer1/lstm_cell_14/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*;
shared_name,*price_layer1/lstm_cell_14/recurrent_kernel
?
>price_layer1/lstm_cell_14/recurrent_kernel/Read/ReadVariableOpReadVariableOp*price_layer1/lstm_cell_14/recurrent_kernel*
_output_shapes
:	 ?*
dtype0
?
price_layer1/lstm_cell_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name price_layer1/lstm_cell_14/bias
?
2price_layer1/lstm_cell_14/bias/Read/ReadVariableOpReadVariableOpprice_layer1/lstm_cell_14/bias*
_output_shapes	
:?*
dtype0
?
 price_layer2/lstm_cell_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*1
shared_name" price_layer2/lstm_cell_15/kernel
?
4price_layer2/lstm_cell_15/kernel/Read/ReadVariableOpReadVariableOp price_layer2/lstm_cell_15/kernel*
_output_shapes
:	 ?*
dtype0
?
*price_layer2/lstm_cell_15/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*;
shared_name,*price_layer2/lstm_cell_15/recurrent_kernel
?
>price_layer2/lstm_cell_15/recurrent_kernel/Read/ReadVariableOpReadVariableOp*price_layer2/lstm_cell_15/recurrent_kernel*
_output_shapes
:	 ?*
dtype0
?
price_layer2/lstm_cell_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name price_layer2/lstm_cell_15/bias
?
2price_layer2/lstm_cell_15/bias/Read/ReadVariableOpReadVariableOpprice_layer2/lstm_cell_15/bias*
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
'Adam/price_layer1/lstm_cell_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'Adam/price_layer1/lstm_cell_14/kernel/m
?
;Adam/price_layer1/lstm_cell_14/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/price_layer1/lstm_cell_14/kernel/m*
_output_shapes
:	?*
dtype0
?
1Adam/price_layer1/lstm_cell_14/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*B
shared_name31Adam/price_layer1/lstm_cell_14/recurrent_kernel/m
?
EAdam/price_layer1/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp1Adam/price_layer1/lstm_cell_14/recurrent_kernel/m*
_output_shapes
:	 ?*
dtype0
?
%Adam/price_layer1/lstm_cell_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/price_layer1/lstm_cell_14/bias/m
?
9Adam/price_layer1/lstm_cell_14/bias/m/Read/ReadVariableOpReadVariableOp%Adam/price_layer1/lstm_cell_14/bias/m*
_output_shapes	
:?*
dtype0
?
'Adam/price_layer2/lstm_cell_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*8
shared_name)'Adam/price_layer2/lstm_cell_15/kernel/m
?
;Adam/price_layer2/lstm_cell_15/kernel/m/Read/ReadVariableOpReadVariableOp'Adam/price_layer2/lstm_cell_15/kernel/m*
_output_shapes
:	 ?*
dtype0
?
1Adam/price_layer2/lstm_cell_15/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*B
shared_name31Adam/price_layer2/lstm_cell_15/recurrent_kernel/m
?
EAdam/price_layer2/lstm_cell_15/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp1Adam/price_layer2/lstm_cell_15/recurrent_kernel/m*
_output_shapes
:	 ?*
dtype0
?
%Adam/price_layer2/lstm_cell_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/price_layer2/lstm_cell_15/bias/m
?
9Adam/price_layer2/lstm_cell_15/bias/m/Read/ReadVariableOpReadVariableOp%Adam/price_layer2/lstm_cell_15/bias/m*
_output_shapes	
:?*
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
?
'Adam/price_layer1/lstm_cell_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'Adam/price_layer1/lstm_cell_14/kernel/v
?
;Adam/price_layer1/lstm_cell_14/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/price_layer1/lstm_cell_14/kernel/v*
_output_shapes
:	?*
dtype0
?
1Adam/price_layer1/lstm_cell_14/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*B
shared_name31Adam/price_layer1/lstm_cell_14/recurrent_kernel/v
?
EAdam/price_layer1/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp1Adam/price_layer1/lstm_cell_14/recurrent_kernel/v*
_output_shapes
:	 ?*
dtype0
?
%Adam/price_layer1/lstm_cell_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/price_layer1/lstm_cell_14/bias/v
?
9Adam/price_layer1/lstm_cell_14/bias/v/Read/ReadVariableOpReadVariableOp%Adam/price_layer1/lstm_cell_14/bias/v*
_output_shapes	
:?*
dtype0
?
'Adam/price_layer2/lstm_cell_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*8
shared_name)'Adam/price_layer2/lstm_cell_15/kernel/v
?
;Adam/price_layer2/lstm_cell_15/kernel/v/Read/ReadVariableOpReadVariableOp'Adam/price_layer2/lstm_cell_15/kernel/v*
_output_shapes
:	 ?*
dtype0
?
1Adam/price_layer2/lstm_cell_15/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*B
shared_name31Adam/price_layer2/lstm_cell_15/recurrent_kernel/v
?
EAdam/price_layer2/lstm_cell_15/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp1Adam/price_layer2/lstm_cell_15/recurrent_kernel/v*
_output_shapes
:	 ?*
dtype0
?
%Adam/price_layer2/lstm_cell_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%Adam/price_layer2/lstm_cell_15/bias/v
?
9Adam/price_layer2/lstm_cell_15/bias/v/Read/ReadVariableOpReadVariableOp%Adam/price_layer2/lstm_cell_15/bias/v*
_output_shapes	
:?*
dtype0

NoOpNoOp
?G
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?G
value?GB?F B?F
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
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate%m?&m?+m?,m?1m?2m?<m?=m?>m??m?@m?Am?%v?&v?+v?,v?1v?2v?<v?=v?>v??v?@v?Av?
 
 
V
<0
=1
>2
?3
@4
A5
%6
&7
+8
,9
110
211
V
<0
=1
>2
?3
@4
A5
%6
&7
+8
,9
110
211
?
Bmetrics
Clayer_regularization_losses

Dlayers
Enon_trainable_variables
regularization_losses
	variables
Flayer_metrics
trainable_variables
 
~

<kernel
=recurrent_kernel
>bias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
 
 

<0
=1
>2

<0
=1
>2
?
Kmetrics
Llayer_regularization_losses

Mlayers
Nnon_trainable_variables
regularization_losses
trainable_variables
	variables
Olayer_metrics

Pstates
~

?kernel
@recurrent_kernel
Abias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
 
 

?0
@1
A2

?0
@1
A2
?
Umetrics
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
regularization_losses
trainable_variables
	variables
Ylayer_metrics

Zstates
 
 
 
?
[layer_regularization_losses
\metrics

]layers
^non_trainable_variables
regularization_losses
	variables
_layer_metrics
trainable_variables
 
 
 
?
`layer_regularization_losses
ametrics

blayers
cnon_trainable_variables
!regularization_losses
"	variables
dlayer_metrics
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
elayer_regularization_losses
fmetrics

glayers
hnon_trainable_variables
'regularization_losses
(	variables
ilayer_metrics
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
jlayer_regularization_losses
kmetrics

llayers
mnon_trainable_variables
-regularization_losses
.	variables
nlayer_metrics
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
olayer_regularization_losses
pmetrics

qlayers
rnon_trainable_variables
3regularization_losses
4	variables
slayer_metrics
5trainable_variables
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
\Z
VARIABLE_VALUE price_layer1/lstm_cell_14/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*price_layer1/lstm_cell_14/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEprice_layer1/lstm_cell_14/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE price_layer2/lstm_cell_15/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*price_layer2/lstm_cell_15/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEprice_layer2/lstm_cell_15/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

t0
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
<0
=1
>2

<0
=1
>2
?
ulayer_regularization_losses
vmetrics

wlayers
xnon_trainable_variables
Gregularization_losses
H	variables
ylayer_metrics
Itrainable_variables
 
 

0
 
 
 
 

?0
@1
A2

?0
@1
A2
?
zlayer_regularization_losses
{metrics

|layers
}non_trainable_variables
Qregularization_losses
R	variables
~layer_metrics
Strainable_variables
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
7
	total

?count
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

0
?1

?	variables
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
}
VARIABLE_VALUE'Adam/price_layer1/lstm_cell_14/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/price_layer1/lstm_cell_14/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/price_layer1/lstm_cell_14/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/price_layer2/lstm_cell_15/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/price_layer2/lstm_cell_15/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/price_layer2/lstm_cell_15/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
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
}
VARIABLE_VALUE'Adam/price_layer1/lstm_cell_14/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/price_layer1/lstm_cell_14/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/price_layer1/lstm_cell_14/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/price_layer2/lstm_cell_15/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adam/price_layer2/lstm_cell_15/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/price_layer2/lstm_cell_15/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_env_inputserving_default_price_input price_layer1/lstm_cell_14/kernel*price_layer1/lstm_cell_14/recurrent_kernelprice_layer1/lstm_cell_14/bias price_layer2/lstm_cell_15/kernel*price_layer2/lstm_cell_15/recurrent_kernelprice_layer2/lstm_cell_15/biasfixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias*
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
GPU 2J 8? *1
f,R*
(__inference_signature_wrapper_1196146636
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'fixed_layer1/kernel/Read/ReadVariableOp%fixed_layer1/bias/Read/ReadVariableOp'fixed_layer2/kernel/Read/ReadVariableOp%fixed_layer2/bias/Read/ReadVariableOp(action_output/kernel/Read/ReadVariableOp&action_output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp4price_layer1/lstm_cell_14/kernel/Read/ReadVariableOp>price_layer1/lstm_cell_14/recurrent_kernel/Read/ReadVariableOp2price_layer1/lstm_cell_14/bias/Read/ReadVariableOp4price_layer2/lstm_cell_15/kernel/Read/ReadVariableOp>price_layer2/lstm_cell_15/recurrent_kernel/Read/ReadVariableOp2price_layer2/lstm_cell_15/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/fixed_layer1/kernel/m/Read/ReadVariableOp,Adam/fixed_layer1/bias/m/Read/ReadVariableOp.Adam/fixed_layer2/kernel/m/Read/ReadVariableOp,Adam/fixed_layer2/bias/m/Read/ReadVariableOp/Adam/action_output/kernel/m/Read/ReadVariableOp-Adam/action_output/bias/m/Read/ReadVariableOp;Adam/price_layer1/lstm_cell_14/kernel/m/Read/ReadVariableOpEAdam/price_layer1/lstm_cell_14/recurrent_kernel/m/Read/ReadVariableOp9Adam/price_layer1/lstm_cell_14/bias/m/Read/ReadVariableOp;Adam/price_layer2/lstm_cell_15/kernel/m/Read/ReadVariableOpEAdam/price_layer2/lstm_cell_15/recurrent_kernel/m/Read/ReadVariableOp9Adam/price_layer2/lstm_cell_15/bias/m/Read/ReadVariableOp.Adam/fixed_layer1/kernel/v/Read/ReadVariableOp,Adam/fixed_layer1/bias/v/Read/ReadVariableOp.Adam/fixed_layer2/kernel/v/Read/ReadVariableOp,Adam/fixed_layer2/bias/v/Read/ReadVariableOp/Adam/action_output/kernel/v/Read/ReadVariableOp-Adam/action_output/bias/v/Read/ReadVariableOp;Adam/price_layer1/lstm_cell_14/kernel/v/Read/ReadVariableOpEAdam/price_layer1/lstm_cell_14/recurrent_kernel/v/Read/ReadVariableOp9Adam/price_layer1/lstm_cell_14/bias/v/Read/ReadVariableOp;Adam/price_layer2/lstm_cell_15/kernel/v/Read/ReadVariableOpEAdam/price_layer2/lstm_cell_15/recurrent_kernel/v/Read/ReadVariableOp9Adam/price_layer2/lstm_cell_15/bias/v/Read/ReadVariableOpConst*8
Tin1
/2-	*
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
#__inference__traced_save_1196149098
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefixed_layer1/kernelfixed_layer1/biasfixed_layer2/kernelfixed_layer2/biasaction_output/kernelaction_output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate price_layer1/lstm_cell_14/kernel*price_layer1/lstm_cell_14/recurrent_kernelprice_layer1/lstm_cell_14/bias price_layer2/lstm_cell_15/kernel*price_layer2/lstm_cell_15/recurrent_kernelprice_layer2/lstm_cell_15/biastotalcountAdam/fixed_layer1/kernel/mAdam/fixed_layer1/bias/mAdam/fixed_layer2/kernel/mAdam/fixed_layer2/bias/mAdam/action_output/kernel/mAdam/action_output/bias/m'Adam/price_layer1/lstm_cell_14/kernel/m1Adam/price_layer1/lstm_cell_14/recurrent_kernel/m%Adam/price_layer1/lstm_cell_14/bias/m'Adam/price_layer2/lstm_cell_15/kernel/m1Adam/price_layer2/lstm_cell_15/recurrent_kernel/m%Adam/price_layer2/lstm_cell_15/bias/mAdam/fixed_layer1/kernel/vAdam/fixed_layer1/bias/vAdam/fixed_layer2/kernel/vAdam/fixed_layer2/bias/vAdam/action_output/kernel/vAdam/action_output/bias/v'Adam/price_layer1/lstm_cell_14/kernel/v1Adam/price_layer1/lstm_cell_14/recurrent_kernel/v%Adam/price_layer1/lstm_cell_14/bias/v'Adam/price_layer2/lstm_cell_15/kernel/v1Adam/price_layer2/lstm_cell_15/recurrent_kernel/v%Adam/price_layer2/lstm_cell_15/bias/v*7
Tin0
.2,*
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
&__inference__traced_restore_1196149237??%
?
?
while_cond_1196146047
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196146047___redundant_placeholder08
4while_while_cond_1196146047___redundant_placeholder18
4while_while_cond_1196146047___redundant_placeholder28
4while_while_cond_1196146047___redundant_placeholder3
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
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196144527

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
:	?*
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
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147503
inputs_0/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileF
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196147418*!
condR
while_cond_1196147417*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?	
?
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1196148697

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
?_
?
#__inference__traced_save_1196149098
file_prefix2
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
-savev2_adam_learning_rate_read_readvariableop?
;savev2_price_layer1_lstm_cell_14_kernel_read_readvariableopI
Esavev2_price_layer1_lstm_cell_14_recurrent_kernel_read_readvariableop=
9savev2_price_layer1_lstm_cell_14_bias_read_readvariableop?
;savev2_price_layer2_lstm_cell_15_kernel_read_readvariableopI
Esavev2_price_layer2_lstm_cell_15_recurrent_kernel_read_readvariableop=
9savev2_price_layer2_lstm_cell_15_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_fixed_layer1_kernel_m_read_readvariableop7
3savev2_adam_fixed_layer1_bias_m_read_readvariableop9
5savev2_adam_fixed_layer2_kernel_m_read_readvariableop7
3savev2_adam_fixed_layer2_bias_m_read_readvariableop:
6savev2_adam_action_output_kernel_m_read_readvariableop8
4savev2_adam_action_output_bias_m_read_readvariableopF
Bsavev2_adam_price_layer1_lstm_cell_14_kernel_m_read_readvariableopP
Lsavev2_adam_price_layer1_lstm_cell_14_recurrent_kernel_m_read_readvariableopD
@savev2_adam_price_layer1_lstm_cell_14_bias_m_read_readvariableopF
Bsavev2_adam_price_layer2_lstm_cell_15_kernel_m_read_readvariableopP
Lsavev2_adam_price_layer2_lstm_cell_15_recurrent_kernel_m_read_readvariableopD
@savev2_adam_price_layer2_lstm_cell_15_bias_m_read_readvariableop9
5savev2_adam_fixed_layer1_kernel_v_read_readvariableop7
3savev2_adam_fixed_layer1_bias_v_read_readvariableop9
5savev2_adam_fixed_layer2_kernel_v_read_readvariableop7
3savev2_adam_fixed_layer2_bias_v_read_readvariableop:
6savev2_adam_action_output_kernel_v_read_readvariableop8
4savev2_adam_action_output_bias_v_read_readvariableopF
Bsavev2_adam_price_layer1_lstm_cell_14_kernel_v_read_readvariableopP
Lsavev2_adam_price_layer1_lstm_cell_14_recurrent_kernel_v_read_readvariableopD
@savev2_adam_price_layer1_lstm_cell_14_bias_v_read_readvariableopF
Bsavev2_adam_price_layer2_lstm_cell_15_kernel_v_read_readvariableopP
Lsavev2_adam_price_layer2_lstm_cell_15_recurrent_kernel_v_read_readvariableopD
@savev2_adam_price_layer2_lstm_cell_15_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_fixed_layer1_kernel_read_readvariableop,savev2_fixed_layer1_bias_read_readvariableop.savev2_fixed_layer2_kernel_read_readvariableop,savev2_fixed_layer2_bias_read_readvariableop/savev2_action_output_kernel_read_readvariableop-savev2_action_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop;savev2_price_layer1_lstm_cell_14_kernel_read_readvariableopEsavev2_price_layer1_lstm_cell_14_recurrent_kernel_read_readvariableop9savev2_price_layer1_lstm_cell_14_bias_read_readvariableop;savev2_price_layer2_lstm_cell_15_kernel_read_readvariableopEsavev2_price_layer2_lstm_cell_15_recurrent_kernel_read_readvariableop9savev2_price_layer2_lstm_cell_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_fixed_layer1_kernel_m_read_readvariableop3savev2_adam_fixed_layer1_bias_m_read_readvariableop5savev2_adam_fixed_layer2_kernel_m_read_readvariableop3savev2_adam_fixed_layer2_bias_m_read_readvariableop6savev2_adam_action_output_kernel_m_read_readvariableop4savev2_adam_action_output_bias_m_read_readvariableopBsavev2_adam_price_layer1_lstm_cell_14_kernel_m_read_readvariableopLsavev2_adam_price_layer1_lstm_cell_14_recurrent_kernel_m_read_readvariableop@savev2_adam_price_layer1_lstm_cell_14_bias_m_read_readvariableopBsavev2_adam_price_layer2_lstm_cell_15_kernel_m_read_readvariableopLsavev2_adam_price_layer2_lstm_cell_15_recurrent_kernel_m_read_readvariableop@savev2_adam_price_layer2_lstm_cell_15_bias_m_read_readvariableop5savev2_adam_fixed_layer1_kernel_v_read_readvariableop3savev2_adam_fixed_layer1_bias_v_read_readvariableop5savev2_adam_fixed_layer2_kernel_v_read_readvariableop3savev2_adam_fixed_layer2_bias_v_read_readvariableop6savev2_adam_action_output_kernel_v_read_readvariableop4savev2_adam_action_output_bias_v_read_readvariableopBsavev2_adam_price_layer1_lstm_cell_14_kernel_v_read_readvariableopLsavev2_adam_price_layer1_lstm_cell_14_recurrent_kernel_v_read_readvariableop@savev2_adam_price_layer1_lstm_cell_14_bias_v_read_readvariableopBsavev2_adam_price_layer2_lstm_cell_15_kernel_v_read_readvariableopLsavev2_adam_price_layer2_lstm_cell_15_recurrent_kernel_v_read_readvariableop@savev2_adam_price_layer2_lstm_cell_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *:
dtypes0
.2,	2
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
?: :":::::: : : : : :	?:	 ?:?:	 ?:	 ?:?: : :"::::::	?:	 ?:?:	 ?:	 ?:?:"::::::	?:	 ?:?:	 ?:	 ?:?: 2(
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
: :

_output_shapes
: :%!

_output_shapes
:	?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:%!

_output_shapes
:	 ?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:": 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:%!

_output_shapes
:	 ?:%!

_output_shapes
:	 ?:!

_output_shapes	
:?:$  

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
::%&!

_output_shapes
:	?:%'!

_output_shapes
:	 ?:!(

_output_shapes	
:?:%)!

_output_shapes
:	 ?:%*!

_output_shapes
:	 ?:!+

_output_shapes	
:?:,

_output_shapes
: 
?
?
1__inference_price_layer2_layer_call_fn_1196148323

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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961461332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
%__inference__wrapped_model_1196144421
price_input
	env_inputD
@model_9_price_layer1_lstm_cell_14_matmul_readvariableop_resourceF
Bmodel_9_price_layer1_lstm_cell_14_matmul_1_readvariableop_resourceE
Amodel_9_price_layer1_lstm_cell_14_biasadd_readvariableop_resourceD
@model_9_price_layer2_lstm_cell_15_matmul_readvariableop_resourceF
Bmodel_9_price_layer2_lstm_cell_15_matmul_1_readvariableop_resourceE
Amodel_9_price_layer2_lstm_cell_15_biasadd_readvariableop_resource7
3model_9_fixed_layer1_matmul_readvariableop_resource8
4model_9_fixed_layer1_biasadd_readvariableop_resource7
3model_9_fixed_layer2_matmul_readvariableop_resource8
4model_9_fixed_layer2_biasadd_readvariableop_resource8
4model_9_action_output_matmul_readvariableop_resource9
5model_9_action_output_biasadd_readvariableop_resource
identity??,model_9/action_output/BiasAdd/ReadVariableOp?+model_9/action_output/MatMul/ReadVariableOp?+model_9/fixed_layer1/BiasAdd/ReadVariableOp?*model_9/fixed_layer1/MatMul/ReadVariableOp?+model_9/fixed_layer2/BiasAdd/ReadVariableOp?*model_9/fixed_layer2/MatMul/ReadVariableOp?8model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp?7model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOp?9model_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp?model_9/price_layer1/while?8model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp?7model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOp?9model_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp?model_9/price_layer2/whiles
model_9/price_layer1/ShapeShapeprice_input*
T0*
_output_shapes
:2
model_9/price_layer1/Shape?
(model_9/price_layer1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model_9/price_layer1/strided_slice/stack?
*model_9/price_layer1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_9/price_layer1/strided_slice/stack_1?
*model_9/price_layer1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_9/price_layer1/strided_slice/stack_2?
"model_9/price_layer1/strided_sliceStridedSlice#model_9/price_layer1/Shape:output:01model_9/price_layer1/strided_slice/stack:output:03model_9/price_layer1/strided_slice/stack_1:output:03model_9/price_layer1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model_9/price_layer1/strided_slice?
 model_9/price_layer1/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model_9/price_layer1/zeros/mul/y?
model_9/price_layer1/zeros/mulMul+model_9/price_layer1/strided_slice:output:0)model_9/price_layer1/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
model_9/price_layer1/zeros/mul?
!model_9/price_layer1/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!model_9/price_layer1/zeros/Less/y?
model_9/price_layer1/zeros/LessLess"model_9/price_layer1/zeros/mul:z:0*model_9/price_layer1/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
model_9/price_layer1/zeros/Less?
#model_9/price_layer1/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_9/price_layer1/zeros/packed/1?
!model_9/price_layer1/zeros/packedPack+model_9/price_layer1/strided_slice:output:0,model_9/price_layer1/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!model_9/price_layer1/zeros/packed?
 model_9/price_layer1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_9/price_layer1/zeros/Const?
model_9/price_layer1/zerosFill*model_9/price_layer1/zeros/packed:output:0)model_9/price_layer1/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer1/zeros?
"model_9/price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_9/price_layer1/zeros_1/mul/y?
 model_9/price_layer1/zeros_1/mulMul+model_9/price_layer1/strided_slice:output:0+model_9/price_layer1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 model_9/price_layer1/zeros_1/mul?
#model_9/price_layer1/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_9/price_layer1/zeros_1/Less/y?
!model_9/price_layer1/zeros_1/LessLess$model_9/price_layer1/zeros_1/mul:z:0,model_9/price_layer1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!model_9/price_layer1/zeros_1/Less?
%model_9/price_layer1/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%model_9/price_layer1/zeros_1/packed/1?
#model_9/price_layer1/zeros_1/packedPack+model_9/price_layer1/strided_slice:output:0.model_9/price_layer1/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#model_9/price_layer1/zeros_1/packed?
"model_9/price_layer1/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_9/price_layer1/zeros_1/Const?
model_9/price_layer1/zeros_1Fill,model_9/price_layer1/zeros_1/packed:output:0+model_9/price_layer1/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer1/zeros_1?
#model_9/price_layer1/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model_9/price_layer1/transpose/perm?
model_9/price_layer1/transpose	Transposeprice_input,model_9/price_layer1/transpose/perm:output:0*
T0*+
_output_shapes
:?????????2 
model_9/price_layer1/transpose?
model_9/price_layer1/Shape_1Shape"model_9/price_layer1/transpose:y:0*
T0*
_output_shapes
:2
model_9/price_layer1/Shape_1?
*model_9/price_layer1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_9/price_layer1/strided_slice_1/stack?
,model_9/price_layer1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer1/strided_slice_1/stack_1?
,model_9/price_layer1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer1/strided_slice_1/stack_2?
$model_9/price_layer1/strided_slice_1StridedSlice%model_9/price_layer1/Shape_1:output:03model_9/price_layer1/strided_slice_1/stack:output:05model_9/price_layer1/strided_slice_1/stack_1:output:05model_9/price_layer1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model_9/price_layer1/strided_slice_1?
0model_9/price_layer1/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model_9/price_layer1/TensorArrayV2/element_shape?
"model_9/price_layer1/TensorArrayV2TensorListReserve9model_9/price_layer1/TensorArrayV2/element_shape:output:0-model_9/price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model_9/price_layer1/TensorArrayV2?
Jmodel_9/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2L
Jmodel_9/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape?
<model_9/price_layer1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_9/price_layer1/transpose:y:0Smodel_9/price_layer1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<model_9/price_layer1/TensorArrayUnstack/TensorListFromTensor?
*model_9/price_layer1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_9/price_layer1/strided_slice_2/stack?
,model_9/price_layer1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer1/strided_slice_2/stack_1?
,model_9/price_layer1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer1/strided_slice_2/stack_2?
$model_9/price_layer1/strided_slice_2StridedSlice"model_9/price_layer1/transpose:y:03model_9/price_layer1/strided_slice_2/stack:output:05model_9/price_layer1/strided_slice_2/stack_1:output:05model_9/price_layer1/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2&
$model_9/price_layer1/strided_slice_2?
7model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@model_9_price_layer1_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype029
7model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOp?
(model_9/price_layer1/lstm_cell_14/MatMulMatMul-model_9/price_layer1/strided_slice_2:output:0?model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_9/price_layer1/lstm_cell_14/MatMul?
9model_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBmodel_9_price_layer1_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02;
9model_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp?
*model_9/price_layer1/lstm_cell_14/MatMul_1MatMul#model_9/price_layer1/zeros:output:0Amodel_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*model_9/price_layer1/lstm_cell_14/MatMul_1?
%model_9/price_layer1/lstm_cell_14/addAddV22model_9/price_layer1/lstm_cell_14/MatMul:product:04model_9/price_layer1/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%model_9/price_layer1/lstm_cell_14/add?
8model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAmodel_9_price_layer1_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp?
)model_9/price_layer1/lstm_cell_14/BiasAddBiasAdd)model_9/price_layer1/lstm_cell_14/add:z:0@model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_9/price_layer1/lstm_cell_14/BiasAdd?
'model_9/price_layer1/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_9/price_layer1/lstm_cell_14/Const?
1model_9/price_layer1/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1model_9/price_layer1/lstm_cell_14/split/split_dim?
'model_9/price_layer1/lstm_cell_14/splitSplit:model_9/price_layer1/lstm_cell_14/split/split_dim:output:02model_9/price_layer1/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2)
'model_9/price_layer1/lstm_cell_14/split?
)model_9/price_layer1/lstm_cell_14/SigmoidSigmoid0model_9/price_layer1/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2+
)model_9/price_layer1/lstm_cell_14/Sigmoid?
+model_9/price_layer1/lstm_cell_14/Sigmoid_1Sigmoid0model_9/price_layer1/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2-
+model_9/price_layer1/lstm_cell_14/Sigmoid_1?
%model_9/price_layer1/lstm_cell_14/mulMul/model_9/price_layer1/lstm_cell_14/Sigmoid_1:y:0%model_9/price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2'
%model_9/price_layer1/lstm_cell_14/mul?
&model_9/price_layer1/lstm_cell_14/ReluRelu0model_9/price_layer1/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2(
&model_9/price_layer1/lstm_cell_14/Relu?
'model_9/price_layer1/lstm_cell_14/mul_1Mul-model_9/price_layer1/lstm_cell_14/Sigmoid:y:04model_9/price_layer1/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2)
'model_9/price_layer1/lstm_cell_14/mul_1?
'model_9/price_layer1/lstm_cell_14/add_1AddV2)model_9/price_layer1/lstm_cell_14/mul:z:0+model_9/price_layer1/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2)
'model_9/price_layer1/lstm_cell_14/add_1?
+model_9/price_layer1/lstm_cell_14/Sigmoid_2Sigmoid0model_9/price_layer1/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2-
+model_9/price_layer1/lstm_cell_14/Sigmoid_2?
(model_9/price_layer1/lstm_cell_14/Relu_1Relu+model_9/price_layer1/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2*
(model_9/price_layer1/lstm_cell_14/Relu_1?
'model_9/price_layer1/lstm_cell_14/mul_2Mul/model_9/price_layer1/lstm_cell_14/Sigmoid_2:y:06model_9/price_layer1/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2)
'model_9/price_layer1/lstm_cell_14/mul_2?
2model_9/price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    24
2model_9/price_layer1/TensorArrayV2_1/element_shape?
$model_9/price_layer1/TensorArrayV2_1TensorListReserve;model_9/price_layer1/TensorArrayV2_1/element_shape:output:0-model_9/price_layer1/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$model_9/price_layer1/TensorArrayV2_1x
model_9/price_layer1/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/price_layer1/time?
-model_9/price_layer1/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model_9/price_layer1/while/maximum_iterations?
'model_9/price_layer1/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_9/price_layer1/while/loop_counter?
model_9/price_layer1/whileWhile0model_9/price_layer1/while/loop_counter:output:06model_9/price_layer1/while/maximum_iterations:output:0"model_9/price_layer1/time:output:0-model_9/price_layer1/TensorArrayV2_1:handle:0#model_9/price_layer1/zeros:output:0%model_9/price_layer1/zeros_1:output:0-model_9/price_layer1/strided_slice_1:output:0Lmodel_9/price_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:0@model_9_price_layer1_lstm_cell_14_matmul_readvariableop_resourceBmodel_9_price_layer1_lstm_cell_14_matmul_1_readvariableop_resourceAmodel_9_price_layer1_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*6
body.R,
*model_9_price_layer1_while_body_1196144163*6
cond.R,
*model_9_price_layer1_while_cond_1196144162*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
model_9/price_layer1/while?
Emodel_9/price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2G
Emodel_9/price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
7model_9/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStack#model_9/price_layer1/while:output:3Nmodel_9/price_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype029
7model_9/price_layer1/TensorArrayV2Stack/TensorListStack?
*model_9/price_layer1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*model_9/price_layer1/strided_slice_3/stack?
,model_9/price_layer1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,model_9/price_layer1/strided_slice_3/stack_1?
,model_9/price_layer1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer1/strided_slice_3/stack_2?
$model_9/price_layer1/strided_slice_3StridedSlice@model_9/price_layer1/TensorArrayV2Stack/TensorListStack:tensor:03model_9/price_layer1/strided_slice_3/stack:output:05model_9/price_layer1/strided_slice_3/stack_1:output:05model_9/price_layer1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2&
$model_9/price_layer1/strided_slice_3?
%model_9/price_layer1/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%model_9/price_layer1/transpose_1/perm?
 model_9/price_layer1/transpose_1	Transpose@model_9/price_layer1/TensorArrayV2Stack/TensorListStack:tensor:0.model_9/price_layer1/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2"
 model_9/price_layer1/transpose_1?
model_9/price_layer1/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/price_layer1/runtime?
model_9/price_layer2/ShapeShape$model_9/price_layer1/transpose_1:y:0*
T0*
_output_shapes
:2
model_9/price_layer2/Shape?
(model_9/price_layer2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(model_9/price_layer2/strided_slice/stack?
*model_9/price_layer2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_9/price_layer2/strided_slice/stack_1?
*model_9/price_layer2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*model_9/price_layer2/strided_slice/stack_2?
"model_9/price_layer2/strided_sliceStridedSlice#model_9/price_layer2/Shape:output:01model_9/price_layer2/strided_slice/stack:output:03model_9/price_layer2/strided_slice/stack_1:output:03model_9/price_layer2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"model_9/price_layer2/strided_slice?
 model_9/price_layer2/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2"
 model_9/price_layer2/zeros/mul/y?
model_9/price_layer2/zeros/mulMul+model_9/price_layer2/strided_slice:output:0)model_9/price_layer2/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
model_9/price_layer2/zeros/mul?
!model_9/price_layer2/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2#
!model_9/price_layer2/zeros/Less/y?
model_9/price_layer2/zeros/LessLess"model_9/price_layer2/zeros/mul:z:0*model_9/price_layer2/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
model_9/price_layer2/zeros/Less?
#model_9/price_layer2/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2%
#model_9/price_layer2/zeros/packed/1?
!model_9/price_layer2/zeros/packedPack+model_9/price_layer2/strided_slice:output:0,model_9/price_layer2/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!model_9/price_layer2/zeros/packed?
 model_9/price_layer2/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_9/price_layer2/zeros/Const?
model_9/price_layer2/zerosFill*model_9/price_layer2/zeros/packed:output:0)model_9/price_layer2/zeros/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer2/zeros?
"model_9/price_layer2/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2$
"model_9/price_layer2/zeros_1/mul/y?
 model_9/price_layer2/zeros_1/mulMul+model_9/price_layer2/strided_slice:output:0+model_9/price_layer2/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 model_9/price_layer2/zeros_1/mul?
#model_9/price_layer2/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2%
#model_9/price_layer2/zeros_1/Less/y?
!model_9/price_layer2/zeros_1/LessLess$model_9/price_layer2/zeros_1/mul:z:0,model_9/price_layer2/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!model_9/price_layer2/zeros_1/Less?
%model_9/price_layer2/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B : 2'
%model_9/price_layer2/zeros_1/packed/1?
#model_9/price_layer2/zeros_1/packedPack+model_9/price_layer2/strided_slice:output:0.model_9/price_layer2/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#model_9/price_layer2/zeros_1/packed?
"model_9/price_layer2/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_9/price_layer2/zeros_1/Const?
model_9/price_layer2/zeros_1Fill,model_9/price_layer2/zeros_1/packed:output:0+model_9/price_layer2/zeros_1/Const:output:0*
T0*'
_output_shapes
:????????? 2
model_9/price_layer2/zeros_1?
#model_9/price_layer2/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#model_9/price_layer2/transpose/perm?
model_9/price_layer2/transpose	Transpose$model_9/price_layer1/transpose_1:y:0,model_9/price_layer2/transpose/perm:output:0*
T0*+
_output_shapes
:????????? 2 
model_9/price_layer2/transpose?
model_9/price_layer2/Shape_1Shape"model_9/price_layer2/transpose:y:0*
T0*
_output_shapes
:2
model_9/price_layer2/Shape_1?
*model_9/price_layer2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_9/price_layer2/strided_slice_1/stack?
,model_9/price_layer2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer2/strided_slice_1/stack_1?
,model_9/price_layer2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer2/strided_slice_1/stack_2?
$model_9/price_layer2/strided_slice_1StridedSlice%model_9/price_layer2/Shape_1:output:03model_9/price_layer2/strided_slice_1/stack:output:05model_9/price_layer2/strided_slice_1/stack_1:output:05model_9/price_layer2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model_9/price_layer2/strided_slice_1?
0model_9/price_layer2/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????22
0model_9/price_layer2/TensorArrayV2/element_shape?
"model_9/price_layer2/TensorArrayV2TensorListReserve9model_9/price_layer2/TensorArrayV2/element_shape:output:0-model_9/price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"model_9/price_layer2/TensorArrayV2?
Jmodel_9/price_layer2/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2L
Jmodel_9/price_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape?
<model_9/price_layer2/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"model_9/price_layer2/transpose:y:0Smodel_9/price_layer2/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<model_9/price_layer2/TensorArrayUnstack/TensorListFromTensor?
*model_9/price_layer2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model_9/price_layer2/strided_slice_2/stack?
,model_9/price_layer2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer2/strided_slice_2/stack_1?
,model_9/price_layer2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer2/strided_slice_2/stack_2?
$model_9/price_layer2/strided_slice_2StridedSlice"model_9/price_layer2/transpose:y:03model_9/price_layer2/strided_slice_2/stack:output:05model_9/price_layer2/strided_slice_2/stack_1:output:05model_9/price_layer2/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2&
$model_9/price_layer2/strided_slice_2?
7model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp@model_9_price_layer2_lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype029
7model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOp?
(model_9/price_layer2/lstm_cell_15/MatMulMatMul-model_9/price_layer2/strided_slice_2:output:0?model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(model_9/price_layer2/lstm_cell_15/MatMul?
9model_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpBmodel_9_price_layer2_lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02;
9model_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp?
*model_9/price_layer2/lstm_cell_15/MatMul_1MatMul#model_9/price_layer2/zeros:output:0Amodel_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2,
*model_9/price_layer2/lstm_cell_15/MatMul_1?
%model_9/price_layer2/lstm_cell_15/addAddV22model_9/price_layer2/lstm_cell_15/MatMul:product:04model_9/price_layer2/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2'
%model_9/price_layer2/lstm_cell_15/add?
8model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpAmodel_9_price_layer2_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp?
)model_9/price_layer2/lstm_cell_15/BiasAddBiasAdd)model_9/price_layer2/lstm_cell_15/add:z:0@model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)model_9/price_layer2/lstm_cell_15/BiasAdd?
'model_9/price_layer2/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2)
'model_9/price_layer2/lstm_cell_15/Const?
1model_9/price_layer2/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1model_9/price_layer2/lstm_cell_15/split/split_dim?
'model_9/price_layer2/lstm_cell_15/splitSplit:model_9/price_layer2/lstm_cell_15/split/split_dim:output:02model_9/price_layer2/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2)
'model_9/price_layer2/lstm_cell_15/split?
)model_9/price_layer2/lstm_cell_15/SigmoidSigmoid0model_9/price_layer2/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2+
)model_9/price_layer2/lstm_cell_15/Sigmoid?
+model_9/price_layer2/lstm_cell_15/Sigmoid_1Sigmoid0model_9/price_layer2/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2-
+model_9/price_layer2/lstm_cell_15/Sigmoid_1?
%model_9/price_layer2/lstm_cell_15/mulMul/model_9/price_layer2/lstm_cell_15/Sigmoid_1:y:0%model_9/price_layer2/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2'
%model_9/price_layer2/lstm_cell_15/mul?
&model_9/price_layer2/lstm_cell_15/ReluRelu0model_9/price_layer2/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2(
&model_9/price_layer2/lstm_cell_15/Relu?
'model_9/price_layer2/lstm_cell_15/mul_1Mul-model_9/price_layer2/lstm_cell_15/Sigmoid:y:04model_9/price_layer2/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2)
'model_9/price_layer2/lstm_cell_15/mul_1?
'model_9/price_layer2/lstm_cell_15/add_1AddV2)model_9/price_layer2/lstm_cell_15/mul:z:0+model_9/price_layer2/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2)
'model_9/price_layer2/lstm_cell_15/add_1?
+model_9/price_layer2/lstm_cell_15/Sigmoid_2Sigmoid0model_9/price_layer2/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2-
+model_9/price_layer2/lstm_cell_15/Sigmoid_2?
(model_9/price_layer2/lstm_cell_15/Relu_1Relu+model_9/price_layer2/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2*
(model_9/price_layer2/lstm_cell_15/Relu_1?
'model_9/price_layer2/lstm_cell_15/mul_2Mul/model_9/price_layer2/lstm_cell_15/Sigmoid_2:y:06model_9/price_layer2/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2)
'model_9/price_layer2/lstm_cell_15/mul_2?
2model_9/price_layer2/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    24
2model_9/price_layer2/TensorArrayV2_1/element_shape?
$model_9/price_layer2/TensorArrayV2_1TensorListReserve;model_9/price_layer2/TensorArrayV2_1/element_shape:output:0-model_9/price_layer2/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02&
$model_9/price_layer2/TensorArrayV2_1x
model_9/price_layer2/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
model_9/price_layer2/time?
-model_9/price_layer2/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2/
-model_9/price_layer2/while/maximum_iterations?
'model_9/price_layer2/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_9/price_layer2/while/loop_counter?
model_9/price_layer2/whileWhile0model_9/price_layer2/while/loop_counter:output:06model_9/price_layer2/while/maximum_iterations:output:0"model_9/price_layer2/time:output:0-model_9/price_layer2/TensorArrayV2_1:handle:0#model_9/price_layer2/zeros:output:0%model_9/price_layer2/zeros_1:output:0-model_9/price_layer2/strided_slice_1:output:0Lmodel_9/price_layer2/TensorArrayUnstack/TensorListFromTensor:output_handle:0@model_9_price_layer2_lstm_cell_15_matmul_readvariableop_resourceBmodel_9_price_layer2_lstm_cell_15_matmul_1_readvariableop_resourceAmodel_9_price_layer2_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*6
body.R,
*model_9_price_layer2_while_body_1196144312*6
cond.R,
*model_9_price_layer2_while_cond_1196144311*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
model_9/price_layer2/while?
Emodel_9/price_layer2/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2G
Emodel_9/price_layer2/TensorArrayV2Stack/TensorListStack/element_shape?
7model_9/price_layer2/TensorArrayV2Stack/TensorListStackTensorListStack#model_9/price_layer2/while:output:3Nmodel_9/price_layer2/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
element_dtype029
7model_9/price_layer2/TensorArrayV2Stack/TensorListStack?
*model_9/price_layer2/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2,
*model_9/price_layer2/strided_slice_3/stack?
,model_9/price_layer2/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,model_9/price_layer2/strided_slice_3/stack_1?
,model_9/price_layer2/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model_9/price_layer2/strided_slice_3/stack_2?
$model_9/price_layer2/strided_slice_3StridedSlice@model_9/price_layer2/TensorArrayV2Stack/TensorListStack:tensor:03model_9/price_layer2/strided_slice_3/stack:output:05model_9/price_layer2/strided_slice_3/stack_1:output:05model_9/price_layer2/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:????????? *
shrink_axis_mask2&
$model_9/price_layer2/strided_slice_3?
%model_9/price_layer2/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%model_9/price_layer2/transpose_1/perm?
 model_9/price_layer2/transpose_1	Transpose@model_9/price_layer2/TensorArrayV2Stack/TensorListStack:tensor:0.model_9/price_layer2/transpose_1/perm:output:0*
T0*+
_output_shapes
:????????? 2"
 model_9/price_layer2/transpose_1?
model_9/price_layer2/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
model_9/price_layer2/runtime?
model_9/price_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????    2
model_9/price_flatten/Const?
model_9/price_flatten/ReshapeReshape-model_9/price_layer2/strided_slice_3:output:0$model_9/price_flatten/Const:output:0*
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
model_9/action_output/BiasAdd?
IdentityIdentity&model_9/action_output/BiasAdd:output:0-^model_9/action_output/BiasAdd/ReadVariableOp,^model_9/action_output/MatMul/ReadVariableOp,^model_9/fixed_layer1/BiasAdd/ReadVariableOp+^model_9/fixed_layer1/MatMul/ReadVariableOp,^model_9/fixed_layer2/BiasAdd/ReadVariableOp+^model_9/fixed_layer2/MatMul/ReadVariableOp9^model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp8^model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOp:^model_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp^model_9/price_layer1/while9^model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp8^model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOp:^model_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp^model_9/price_layer2/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:?????????:?????????::::::::::::2\
,model_9/action_output/BiasAdd/ReadVariableOp,model_9/action_output/BiasAdd/ReadVariableOp2Z
+model_9/action_output/MatMul/ReadVariableOp+model_9/action_output/MatMul/ReadVariableOp2Z
+model_9/fixed_layer1/BiasAdd/ReadVariableOp+model_9/fixed_layer1/BiasAdd/ReadVariableOp2X
*model_9/fixed_layer1/MatMul/ReadVariableOp*model_9/fixed_layer1/MatMul/ReadVariableOp2Z
+model_9/fixed_layer2/BiasAdd/ReadVariableOp+model_9/fixed_layer2/BiasAdd/ReadVariableOp2X
*model_9/fixed_layer2/MatMul/ReadVariableOp*model_9/fixed_layer2/MatMul/ReadVariableOp2t
8model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp8model_9/price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp2r
7model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOp7model_9/price_layer1/lstm_cell_14/MatMul/ReadVariableOp2v
9model_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp9model_9/price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp28
model_9/price_layer1/whilemodel_9/price_layer1/while2t
8model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp8model_9/price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp2r
7model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOp7model_9/price_layer2/lstm_cell_15/MatMul/ReadVariableOp2v
9model_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp9model_9/price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp28
model_9/price_layer2/whilemodel_9/price_layer2/while:X T
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
while_cond_1196145865
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196145865___redundant_placeholder08
4while_while_cond_1196145865___redundant_placeholder18
4while_while_cond_1196145865___redundant_placeholder28
4while_while_cond_1196145865___redundant_placeholder3
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
?C
?
while_body_1196147746
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?[
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196146133

inputs/
+lstm_cell_15_matmul_readvariableop_resource1
-lstm_cell_15_matmul_1_readvariableop_resource0
,lstm_cell_15_biasadd_readvariableop_resource
identity??#lstm_cell_15/BiasAdd/ReadVariableOp?"lstm_cell_15/MatMul/ReadVariableOp?$lstm_cell_15/MatMul_1/ReadVariableOp?whileD
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
:????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp?
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul?
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp?
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul_1?
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/add?
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp?
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/BiasAddj
lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/Const~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dim?
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_15/split?
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid?
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_1?
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu?
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_1?
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/add_1?
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu_1?
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196146048*!
condR
while_cond_1196146047*K
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?C
?
while_body_1196146048
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_15_matmul_readvariableop_resource_09
5while_lstm_cell_15_matmul_1_readvariableop_resource_08
4while_lstm_cell_15_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_15_matmul_readvariableop_resource7
3while_lstm_cell_15_matmul_1_readvariableop_resource6
2while_lstm_cell_15_biasadd_readvariableop_resource??)while/lstm_cell_15/BiasAdd/ReadVariableOp?(while/lstm_cell_15/MatMul/ReadVariableOp?*while/lstm_cell_15/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp?
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul?
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp?
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul_1?
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/add?
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp?
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/BiasAddv
while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_15/Const?
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim?
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_15/split?
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid?
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_1?
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul?
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu?
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_1?
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/add_1?
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_2?
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu_1?
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
1__inference_lstm_cell_14_layer_call_fn_1196148828

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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_11961444942
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
E:?????????:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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
?
M__inference_action_output_layer_call_and_return_conditional_losses_1196146410

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
1__inference_lstm_cell_15_layer_call_fn_1196148945

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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_11961451372
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
E:????????? :????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
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
?
?
1__inference_price_layer1_layer_call_fn_1196147667
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
 :?????????????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961448902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

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
?[
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148640
inputs_0/
+lstm_cell_15_matmul_readvariableop_resource1
-lstm_cell_15_matmul_1_readvariableop_resource0
,lstm_cell_15_biasadd_readvariableop_resource
identity??#lstm_cell_15/BiasAdd/ReadVariableOp?"lstm_cell_15/MatMul/ReadVariableOp?$lstm_cell_15/MatMul_1/ReadVariableOp?whileF
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
 :?????????????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp?
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul?
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp?
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul_1?
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/add?
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp?
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/BiasAddj
lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/Const~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dim?
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_15/split?
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid?
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_1?
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu?
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_1?
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/add_1?
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu_1?
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196148555*!
condR
while_cond_1196148554*K
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????????????? :::2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?
?
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196148811

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
:	?*
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
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
?[
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147656
inputs_0/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileF
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196147571*!
condR
while_cond_1196147570*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :??????????????????
"
_user_specified_name
inputs/0
?C
?
while_body_1196148555
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_15_matmul_readvariableop_resource_09
5while_lstm_cell_15_matmul_1_readvariableop_resource_08
4while_lstm_cell_15_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_15_matmul_readvariableop_resource7
3while_lstm_cell_15_matmul_1_readvariableop_resource6
2while_lstm_cell_15_biasadd_readvariableop_resource??)while/lstm_cell_15/BiasAdd/ReadVariableOp?(while/lstm_cell_15/MatMul/ReadVariableOp?*while/lstm_cell_15/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp?
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul?
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp?
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul_1?
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/add?
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp?
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/BiasAddv
while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_15/Const?
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim?
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_15/split?
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid?
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_1?
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul?
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu?
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_1?
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/add_1?
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_2?
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu_1?
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
?%
?
while_body_1196145563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_lstm_cell_15_1196145587_0#
while_lstm_cell_15_1196145589_0#
while_lstm_cell_15_1196145591_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_lstm_cell_15_1196145587!
while_lstm_cell_15_1196145589!
while_lstm_cell_15_1196145591??*while/lstm_cell_15/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_1196145587_0while_lstm_cell_15_1196145589_0while_lstm_cell_15_1196145591_0*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_11961451372,
*while/lstm_cell_15/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_15/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1+^while/lstm_cell_15/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2+^while/lstm_cell_15/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_15_1196145587while_lstm_cell_15_1196145587_0"@
while_lstm_cell_15_1196145589while_lstm_cell_15_1196145589_0"@
while_lstm_cell_15_1196145591while_lstm_cell_15_1196145591_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2X
*while/lstm_cell_15/StatefulPartitionedCall*while/lstm_cell_15/StatefulPartitionedCall: 
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
?[
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148159

inputs/
+lstm_cell_15_matmul_readvariableop_resource1
-lstm_cell_15_matmul_1_readvariableop_resource0
,lstm_cell_15_biasadd_readvariableop_resource
identity??#lstm_cell_15/BiasAdd/ReadVariableOp?"lstm_cell_15/MatMul/ReadVariableOp?$lstm_cell_15/MatMul_1/ReadVariableOp?whileD
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
:????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp?
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul?
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp?
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul_1?
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/add?
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp?
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/BiasAddj
lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/Const~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dim?
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_15/split?
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid?
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_1?
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu?
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_1?
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/add_1?
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu_1?
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196148074*!
condR
while_cond_1196148073*K
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?[
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148312

inputs/
+lstm_cell_15_matmul_readvariableop_resource1
-lstm_cell_15_matmul_1_readvariableop_resource0
,lstm_cell_15_biasadd_readvariableop_resource
identity??#lstm_cell_15/BiasAdd/ReadVariableOp?"lstm_cell_15/MatMul/ReadVariableOp?$lstm_cell_15/MatMul_1/ReadVariableOp?whileD
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
:????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp?
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul?
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp?
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul_1?
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/add?
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp?
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/BiasAddj
lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/Const~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dim?
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_15/split?
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid?
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_1?
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu?
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_1?
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/add_1?
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu_1?
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196148227*!
condR
while_cond_1196148226*K
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?C
?
while_body_1196148402
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_15_matmul_readvariableop_resource_09
5while_lstm_cell_15_matmul_1_readvariableop_resource_08
4while_lstm_cell_15_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_15_matmul_readvariableop_resource7
3while_lstm_cell_15_matmul_1_readvariableop_resource6
2while_lstm_cell_15_biasadd_readvariableop_resource??)while/lstm_cell_15/BiasAdd/ReadVariableOp?(while/lstm_cell_15/MatMul/ReadVariableOp?*while/lstm_cell_15/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp?
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul?
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp?
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul_1?
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/add?
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp?
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/BiasAddv
while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_15/Const?
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim?
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_15/split?
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid?
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_1?
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul?
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu?
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_1?
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/add_1?
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_2?
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu_1?
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
while_cond_1196145562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196145562___redundant_placeholder08
4while_while_cond_1196145562___redundant_placeholder18
4while_while_cond_1196145562___redundant_placeholder28
4while_while_cond_1196145562___redundant_placeholder3
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
?
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1196146384

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
?
?
while_cond_1196148073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196148073___redundant_placeholder08
4while_while_cond_1196148073___redundant_placeholder18
4while_while_cond_1196148073___redundant_placeholder28
4while_while_cond_1196148073___redundant_placeholder3
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
?[
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147831

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196147746*!
condR
while_cond_1196147745*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?[
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196145798

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196145713*!
condR
while_cond_1196145712*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
,__inference_model_9_layer_call_fn_1196147350
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11961465692
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
?
?
2__inference_action_output_layer_call_fn_1196148745

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
M__inference_action_output_layer_call_and_return_conditional_losses_11961464102
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
?
?
1__inference_price_layer2_layer_call_fn_1196148334

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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961462862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1196148717

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
?&
?
G__inference_model_9_layer_call_and_return_conditional_losses_1196146427
price_input
	env_input
price_layer1_1196145974
price_layer1_1196145976
price_layer1_1196145978
price_layer2_1196146309
price_layer2_1196146311
price_layer2_1196146313
fixed_layer1_1196146368
fixed_layer1_1196146370
fixed_layer2_1196146395
fixed_layer2_1196146397
action_output_1196146421
action_output_1196146423
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_1196145974price_layer1_1196145976price_layer1_1196145978*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961457982&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1196146309price_layer2_1196146311price_layer2_1196146313*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961461332&
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_11961463222
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_11961463372
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1196146368fixed_layer1_1196146370*
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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11961463572&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1196146395fixed_layer2_1196146397*
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11961463842&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1196146421action_output_1196146423*
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
M__inference_action_output_layer_call_and_return_conditional_losses_11961464102'
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
?
i
M__inference_price_flatten_layer_call_and_return_conditional_losses_1196146322

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
1__inference_price_layer1_layer_call_fn_1196148006

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
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961459512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196144890

inputs
lstm_cell_14_1196144808
lstm_cell_14_1196144810
lstm_cell_14_1196144812
identity??$lstm_cell_14/StatefulPartitionedCall?whileD
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
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_1196144808lstm_cell_14_1196144810lstm_cell_14_1196144812*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_11961444942&
$lstm_cell_14/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_1196144808lstm_cell_14_1196144810lstm_cell_14_1196144812*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196144821*!
condR
while_cond_1196144820*K
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
IdentityIdentitytranspose_1:y:0%^lstm_cell_14/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
while_cond_1196147745
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196147745___redundant_placeholder08
4while_while_cond_1196147745___redundant_placeholder18
4while_while_cond_1196147745___redundant_placeholder28
4while_while_cond_1196147745___redundant_placeholder3
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
?%
?
while_body_1196145431
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_lstm_cell_15_1196145455_0#
while_lstm_cell_15_1196145457_0#
while_lstm_cell_15_1196145459_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_lstm_cell_15_1196145455!
while_lstm_cell_15_1196145457!
while_lstm_cell_15_1196145459??*while/lstm_cell_15/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
*while/lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_15_1196145455_0while_lstm_cell_15_1196145457_0while_lstm_cell_15_1196145459_0*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_11961451042,
*while/lstm_cell_15/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_15/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_15/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_15/StatefulPartitionedCall:output:1+^while/lstm_cell_15/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_15/StatefulPartitionedCall:output:2+^while/lstm_cell_15/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_15_1196145455while_lstm_cell_15_1196145455_0"@
while_lstm_cell_15_1196145457while_lstm_cell_15_1196145457_0"@
while_lstm_cell_15_1196145459while_lstm_cell_15_1196145459_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2X
*while/lstm_cell_15/StatefulPartitionedCall*while/lstm_cell_15/StatefulPartitionedCall: 
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
i
M__inference_price_flatten_layer_call_and_return_conditional_losses_1196148668

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
M__inference_action_output_layer_call_and_return_conditional_losses_1196148736

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
?C
?
while_body_1196147899
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196145137

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
:	 ?*
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
E:????????? :????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?C
?
while_body_1196145713
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?[
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148487
inputs_0/
+lstm_cell_15_matmul_readvariableop_resource1
-lstm_cell_15_matmul_1_readvariableop_resource0
,lstm_cell_15_biasadd_readvariableop_resource
identity??#lstm_cell_15/BiasAdd/ReadVariableOp?"lstm_cell_15/MatMul/ReadVariableOp?$lstm_cell_15/MatMul_1/ReadVariableOp?whileF
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
 :?????????????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp?
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul?
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp?
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul_1?
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/add?
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp?
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/BiasAddj
lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/Const~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dim?
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_15/split?
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid?
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_1?
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu?
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_1?
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/add_1?
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu_1?
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196148402*!
condR
while_cond_1196148401*K
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????????????? :::2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?

?
,__inference_model_9_layer_call_fn_1196146530
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11961465032
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
while_cond_1196147570
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196147570___redundant_placeholder08
4while_while_cond_1196147570___redundant_placeholder18
4while_while_cond_1196147570___redundant_placeholder28
4while_while_cond_1196147570___redundant_placeholder3
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
??
?

G__inference_model_9_layer_call_and_return_conditional_losses_1196147290
inputs_0
inputs_1<
8price_layer1_lstm_cell_14_matmul_readvariableop_resource>
:price_layer1_lstm_cell_14_matmul_1_readvariableop_resource=
9price_layer1_lstm_cell_14_biasadd_readvariableop_resource<
8price_layer2_lstm_cell_15_matmul_readvariableop_resource>
:price_layer2_lstm_cell_15_matmul_1_readvariableop_resource=
9price_layer2_lstm_cell_15_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp?/price_layer1/lstm_cell_14/MatMul/ReadVariableOp?1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp?price_layer1/while?0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp?/price_layer2/lstm_cell_15/MatMul/ReadVariableOp?1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp?price_layer2/while`
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
value	B : 2
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
value	B : 2
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
:????????? 2
price_layer1/zerosz
price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2
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
/price_layer1/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp8price_layer1_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/price_layer1/lstm_cell_14/MatMul/ReadVariableOp?
 price_layer1/lstm_cell_14/MatMulMatMul%price_layer1/strided_slice_2:output:07price_layer1/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 price_layer1/lstm_cell_14/MatMul?
1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp:price_layer1_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp?
"price_layer1/lstm_cell_14/MatMul_1MatMulprice_layer1/zeros:output:09price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"price_layer1/lstm_cell_14/MatMul_1?
price_layer1/lstm_cell_14/addAddV2*price_layer1/lstm_cell_14/MatMul:product:0,price_layer1/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
price_layer1/lstm_cell_14/add?
0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp9price_layer1_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp?
!price_layer1/lstm_cell_14/BiasAddBiasAdd!price_layer1/lstm_cell_14/add:z:08price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!price_layer1/lstm_cell_14/BiasAdd?
price_layer1/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
price_layer1/lstm_cell_14/Const?
)price_layer1/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)price_layer1/lstm_cell_14/split/split_dim?
price_layer1/lstm_cell_14/splitSplit2price_layer1/lstm_cell_14/split/split_dim:output:0*price_layer1/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2!
price_layer1/lstm_cell_14/split?
!price_layer1/lstm_cell_14/SigmoidSigmoid(price_layer1/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2#
!price_layer1/lstm_cell_14/Sigmoid?
#price_layer1/lstm_cell_14/Sigmoid_1Sigmoid(price_layer1/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2%
#price_layer1/lstm_cell_14/Sigmoid_1?
price_layer1/lstm_cell_14/mulMul'price_layer1/lstm_cell_14/Sigmoid_1:y:0price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell_14/mul?
price_layer1/lstm_cell_14/ReluRelu(price_layer1/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2 
price_layer1/lstm_cell_14/Relu?
price_layer1/lstm_cell_14/mul_1Mul%price_layer1/lstm_cell_14/Sigmoid:y:0,price_layer1/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_14/mul_1?
price_layer1/lstm_cell_14/add_1AddV2!price_layer1/lstm_cell_14/mul:z:0#price_layer1/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_14/add_1?
#price_layer1/lstm_cell_14/Sigmoid_2Sigmoid(price_layer1/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2%
#price_layer1/lstm_cell_14/Sigmoid_2?
 price_layer1/lstm_cell_14/Relu_1Relu#price_layer1/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 price_layer1/lstm_cell_14/Relu_1?
price_layer1/lstm_cell_14/mul_2Mul'price_layer1/lstm_cell_14/Sigmoid_2:y:0.price_layer1/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_14/mul_2?
*price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2,
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
price_layer1/whileWhile(price_layer1/while/loop_counter:output:0.price_layer1/while/maximum_iterations:output:0price_layer1/time:output:0%price_layer1/TensorArrayV2_1:handle:0price_layer1/zeros:output:0price_layer1/zeros_1:output:0%price_layer1/strided_slice_1:output:0Dprice_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:08price_layer1_lstm_cell_14_matmul_readvariableop_resource:price_layer1_lstm_cell_14_matmul_1_readvariableop_resource9price_layer1_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"price_layer1_while_body_1196147032*.
cond&R$
"price_layer1_while_cond_1196147031*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
price_layer1/while?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStackprice_layer1/while:output:3Fprice_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
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
:????????? *
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
:????????? 2
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
:????????? 2
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
valueB"????    2D
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
:????????? *
shrink_axis_mask2
price_layer2/strided_slice_2?
/price_layer2/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp8price_layer2_lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype021
/price_layer2/lstm_cell_15/MatMul/ReadVariableOp?
 price_layer2/lstm_cell_15/MatMulMatMul%price_layer2/strided_slice_2:output:07price_layer2/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 price_layer2/lstm_cell_15/MatMul?
1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp:price_layer2_lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp?
"price_layer2/lstm_cell_15/MatMul_1MatMulprice_layer2/zeros:output:09price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"price_layer2/lstm_cell_15/MatMul_1?
price_layer2/lstm_cell_15/addAddV2*price_layer2/lstm_cell_15/MatMul:product:0,price_layer2/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
price_layer2/lstm_cell_15/add?
0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp9price_layer2_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp?
!price_layer2/lstm_cell_15/BiasAddBiasAdd!price_layer2/lstm_cell_15/add:z:08price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!price_layer2/lstm_cell_15/BiasAdd?
price_layer2/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
price_layer2/lstm_cell_15/Const?
)price_layer2/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)price_layer2/lstm_cell_15/split/split_dim?
price_layer2/lstm_cell_15/splitSplit2price_layer2/lstm_cell_15/split/split_dim:output:0*price_layer2/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2!
price_layer2/lstm_cell_15/split?
!price_layer2/lstm_cell_15/SigmoidSigmoid(price_layer2/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2#
!price_layer2/lstm_cell_15/Sigmoid?
#price_layer2/lstm_cell_15/Sigmoid_1Sigmoid(price_layer2/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2%
#price_layer2/lstm_cell_15/Sigmoid_1?
price_layer2/lstm_cell_15/mulMul'price_layer2/lstm_cell_15/Sigmoid_1:y:0price_layer2/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/lstm_cell_15/mul?
price_layer2/lstm_cell_15/ReluRelu(price_layer2/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_15/Relu?
price_layer2/lstm_cell_15/mul_1Mul%price_layer2/lstm_cell_15/Sigmoid:y:0,price_layer2/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_15/mul_1?
price_layer2/lstm_cell_15/add_1AddV2!price_layer2/lstm_cell_15/mul:z:0#price_layer2/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_15/add_1?
#price_layer2/lstm_cell_15/Sigmoid_2Sigmoid(price_layer2/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2%
#price_layer2/lstm_cell_15/Sigmoid_2?
 price_layer2/lstm_cell_15/Relu_1Relu#price_layer2/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 price_layer2/lstm_cell_15/Relu_1?
price_layer2/lstm_cell_15/mul_2Mul'price_layer2/lstm_cell_15/Sigmoid_2:y:0.price_layer2/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_15/mul_2?
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
price_layer2/whileWhile(price_layer2/while/loop_counter:output:0.price_layer2/while/maximum_iterations:output:0price_layer2/time:output:0%price_layer2/TensorArrayV2_1:handle:0price_layer2/zeros:output:0price_layer2/zeros_1:output:0%price_layer2/strided_slice_1:output:0Dprice_layer2/TensorArrayUnstack/TensorListFromTensor:output_handle:08price_layer2_lstm_cell_15_matmul_readvariableop_resource:price_layer2_lstm_cell_15_matmul_1_readvariableop_resource9price_layer2_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"price_layer2_while_body_1196147181*.
cond&R$
"price_layer2_while_cond_1196147180*K
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
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp1^price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp0^price_layer1/lstm_cell_14/MatMul/ReadVariableOp2^price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp^price_layer1/while1^price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp0^price_layer2/lstm_cell_15/MatMul/ReadVariableOp2^price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp^price_layer2/while*
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
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2d
0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp2b
/price_layer1/lstm_cell_14/MatMul/ReadVariableOp/price_layer1/lstm_cell_14/MatMul/ReadVariableOp2f
1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp2(
price_layer1/whileprice_layer1/while2d
0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp2b
/price_layer2/lstm_cell_15/MatMul/ReadVariableOp/price_layer2/lstm_cell_15/MatMul/ReadVariableOp2f
1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp2(
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
?
?
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196148911

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
:	 ?*
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
E:????????? :????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
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
?[
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147984

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196147899*!
condR
while_cond_1196147898*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
غ
?
&__inference__traced_restore_1196149237
file_prefix(
$assignvariableop_fixed_layer1_kernel(
$assignvariableop_1_fixed_layer1_bias*
&assignvariableop_2_fixed_layer2_kernel(
$assignvariableop_3_fixed_layer2_bias+
'assignvariableop_4_action_output_kernel)
%assignvariableop_5_action_output_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate8
4assignvariableop_11_price_layer1_lstm_cell_14_kernelB
>assignvariableop_12_price_layer1_lstm_cell_14_recurrent_kernel6
2assignvariableop_13_price_layer1_lstm_cell_14_bias8
4assignvariableop_14_price_layer2_lstm_cell_15_kernelB
>assignvariableop_15_price_layer2_lstm_cell_15_recurrent_kernel6
2assignvariableop_16_price_layer2_lstm_cell_15_bias
assignvariableop_17_total
assignvariableop_18_count2
.assignvariableop_19_adam_fixed_layer1_kernel_m0
,assignvariableop_20_adam_fixed_layer1_bias_m2
.assignvariableop_21_adam_fixed_layer2_kernel_m0
,assignvariableop_22_adam_fixed_layer2_bias_m3
/assignvariableop_23_adam_action_output_kernel_m1
-assignvariableop_24_adam_action_output_bias_m?
;assignvariableop_25_adam_price_layer1_lstm_cell_14_kernel_mI
Eassignvariableop_26_adam_price_layer1_lstm_cell_14_recurrent_kernel_m=
9assignvariableop_27_adam_price_layer1_lstm_cell_14_bias_m?
;assignvariableop_28_adam_price_layer2_lstm_cell_15_kernel_mI
Eassignvariableop_29_adam_price_layer2_lstm_cell_15_recurrent_kernel_m=
9assignvariableop_30_adam_price_layer2_lstm_cell_15_bias_m2
.assignvariableop_31_adam_fixed_layer1_kernel_v0
,assignvariableop_32_adam_fixed_layer1_bias_v2
.assignvariableop_33_adam_fixed_layer2_kernel_v0
,assignvariableop_34_adam_fixed_layer2_bias_v3
/assignvariableop_35_adam_action_output_kernel_v1
-assignvariableop_36_adam_action_output_bias_v?
;assignvariableop_37_adam_price_layer1_lstm_cell_14_kernel_vI
Eassignvariableop_38_adam_price_layer1_lstm_cell_14_recurrent_kernel_v=
9assignvariableop_39_adam_price_layer1_lstm_cell_14_bias_v?
;assignvariableop_40_adam_price_layer2_lstm_cell_15_kernel_vI
Eassignvariableop_41_adam_price_layer2_lstm_cell_15_recurrent_kernel_v=
9assignvariableop_42_adam_price_layer2_lstm_cell_15_bias_v
identity_44??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*?
value?B?,B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	2
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
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp4assignvariableop_11_price_layer1_lstm_cell_14_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp>assignvariableop_12_price_layer1_lstm_cell_14_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp2assignvariableop_13_price_layer1_lstm_cell_14_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp4assignvariableop_14_price_layer2_lstm_cell_15_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp>assignvariableop_15_price_layer2_lstm_cell_15_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_price_layer2_lstm_cell_15_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_fixed_layer1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_fixed_layer1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp.assignvariableop_21_adam_fixed_layer2_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp,assignvariableop_22_adam_fixed_layer2_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_adam_action_output_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp-assignvariableop_24_adam_action_output_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp;assignvariableop_25_adam_price_layer1_lstm_cell_14_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpEassignvariableop_26_adam_price_layer1_lstm_cell_14_recurrent_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp9assignvariableop_27_adam_price_layer1_lstm_cell_14_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp;assignvariableop_28_adam_price_layer2_lstm_cell_15_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpEassignvariableop_29_adam_price_layer2_lstm_cell_15_recurrent_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp9assignvariableop_30_adam_price_layer2_lstm_cell_15_bias_mIdentity_30:output:0"/device:CPU:0*
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
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp;assignvariableop_37_adam_price_layer1_lstm_cell_14_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpEassignvariableop_38_adam_price_layer1_lstm_cell_14_recurrent_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp9assignvariableop_39_adam_price_layer1_lstm_cell_14_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp;assignvariableop_40_adam_price_layer2_lstm_cell_15_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpEassignvariableop_41_adam_price_layer2_lstm_cell_15_recurrent_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp9assignvariableop_42_adam_price_layer2_lstm_cell_15_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_429
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_43Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_43?
Identity_44IdentityIdentity_43:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_44"#
identity_44Identity_44:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422(
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
?
"price_layer2_while_cond_11961468536
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_38
4price_layer2_while_less_price_layer2_strided_slice_1R
Nprice_layer2_while_price_layer2_while_cond_1196146853___redundant_placeholder0R
Nprice_layer2_while_price_layer2_while_cond_1196146853___redundant_placeholder1R
Nprice_layer2_while_price_layer2_while_cond_1196146853___redundant_placeholder2R
Nprice_layer2_while_price_layer2_while_cond_1196146853___redundant_placeholder3
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
?V
?
"price_layer1_while_body_11961470326
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_35
1price_layer1_while_price_layer1_strided_slice_1_0q
mprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0D
@price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0F
Bprice_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0E
Aprice_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0
price_layer1_while_identity!
price_layer1_while_identity_1!
price_layer1_while_identity_2!
price_layer1_while_identity_3!
price_layer1_while_identity_4!
price_layer1_while_identity_53
/price_layer1_while_price_layer1_strided_slice_1o
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensorB
>price_layer1_while_lstm_cell_14_matmul_readvariableop_resourceD
@price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resourceC
?price_layer1_while_lstm_cell_14_biasadd_readvariableop_resource??6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp?5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp?7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp?
&price_layer1/while/lstm_cell_14/MatMulMatMul=price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0=price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&price_layer1/while/lstm_cell_14/MatMul?
7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBprice_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?
(price_layer1/while/lstm_cell_14/MatMul_1MatMul price_layer1_while_placeholder_2?price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(price_layer1/while/lstm_cell_14/MatMul_1?
#price_layer1/while/lstm_cell_14/addAddV20price_layer1/while/lstm_cell_14/MatMul:product:02price_layer1/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#price_layer1/while/lstm_cell_14/add?
6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAprice_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype028
6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp?
'price_layer1/while/lstm_cell_14/BiasAddBiasAdd'price_layer1/while/lstm_cell_14/add:z:0>price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'price_layer1/while/lstm_cell_14/BiasAdd?
%price_layer1/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%price_layer1/while/lstm_cell_14/Const?
/price_layer1/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/price_layer1/while/lstm_cell_14/split/split_dim?
%price_layer1/while/lstm_cell_14/splitSplit8price_layer1/while/lstm_cell_14/split/split_dim:output:00price_layer1/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2'
%price_layer1/while/lstm_cell_14/split?
'price_layer1/while/lstm_cell_14/SigmoidSigmoid.price_layer1/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2)
'price_layer1/while/lstm_cell_14/Sigmoid?
)price_layer1/while/lstm_cell_14/Sigmoid_1Sigmoid.price_layer1/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2+
)price_layer1/while/lstm_cell_14/Sigmoid_1?
#price_layer1/while/lstm_cell_14/mulMul-price_layer1/while/lstm_cell_14/Sigmoid_1:y:0 price_layer1_while_placeholder_3*
T0*'
_output_shapes
:????????? 2%
#price_layer1/while/lstm_cell_14/mul?
$price_layer1/while/lstm_cell_14/ReluRelu.price_layer1/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2&
$price_layer1/while/lstm_cell_14/Relu?
%price_layer1/while/lstm_cell_14/mul_1Mul+price_layer1/while/lstm_cell_14/Sigmoid:y:02price_layer1/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_14/mul_1?
%price_layer1/while/lstm_cell_14/add_1AddV2'price_layer1/while/lstm_cell_14/mul:z:0)price_layer1/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_14/add_1?
)price_layer1/while/lstm_cell_14/Sigmoid_2Sigmoid.price_layer1/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2+
)price_layer1/while/lstm_cell_14/Sigmoid_2?
&price_layer1/while/lstm_cell_14/Relu_1Relu)price_layer1/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2(
&price_layer1/while/lstm_cell_14/Relu_1?
%price_layer1/while/lstm_cell_14/mul_2Mul-price_layer1/while/lstm_cell_14/Sigmoid_2:y:04price_layer1/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_14/mul_2?
7price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer1_while_placeholder_1price_layer1_while_placeholder)price_layer1/while/lstm_cell_14/mul_2:z:0*
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
price_layer1/while/IdentityIdentityprice_layer1/while/add_1:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity?
price_layer1/while/Identity_1Identity8price_layer1_while_price_layer1_while_maximum_iterations7^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_1?
price_layer1/while/Identity_2Identityprice_layer1/while/add:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_2?
price_layer1/while/Identity_3IdentityGprice_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_3?
price_layer1/while/Identity_4Identity)price_layer1/while/lstm_cell_14/mul_2:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer1/while/Identity_4?
price_layer1/while/Identity_5Identity)price_layer1/while/lstm_cell_14/add_1:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer1/while/Identity_5"C
price_layer1_while_identity$price_layer1/while/Identity:output:0"G
price_layer1_while_identity_1&price_layer1/while/Identity_1:output:0"G
price_layer1_while_identity_2&price_layer1/while/Identity_2:output:0"G
price_layer1_while_identity_3&price_layer1/while/Identity_3:output:0"G
price_layer1_while_identity_4&price_layer1/while/Identity_4:output:0"G
price_layer1_while_identity_5&price_layer1/while/Identity_5:output:0"?
?price_layer1_while_lstm_cell_14_biasadd_readvariableop_resourceAprice_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0"?
@price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resourceBprice_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0"?
>price_layer1_while_lstm_cell_14_matmul_readvariableop_resource@price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0"d
/price_layer1_while_price_layer1_strided_slice_11price_layer1_while_price_layer1_strided_slice_1_0"?
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensormprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2p
6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp2n
5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp2r
7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?C
?
while_body_1196145866
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?[
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196145951

inputs/
+lstm_cell_14_matmul_readvariableop_resource1
-lstm_cell_14_matmul_1_readvariableop_resource0
,lstm_cell_14_biasadd_readvariableop_resource
identity??#lstm_cell_14/BiasAdd/ReadVariableOp?"lstm_cell_14/MatMul/ReadVariableOp?$lstm_cell_14/MatMul_1/ReadVariableOp?whileD
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
"lstm_cell_14/MatMul/ReadVariableOpReadVariableOp+lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"lstm_cell_14/MatMul/ReadVariableOp?
lstm_cell_14/MatMulMatMulstrided_slice_2:output:0*lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul?
$lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_14/MatMul_1/ReadVariableOp?
lstm_cell_14/MatMul_1MatMulzeros:output:0,lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/MatMul_1?
lstm_cell_14/addAddV2lstm_cell_14/MatMul:product:0lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/add?
#lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_14/BiasAdd/ReadVariableOp?
lstm_cell_14/BiasAddBiasAddlstm_cell_14/add:z:0+lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_14/BiasAddj
lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/Const~
lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_14/split/split_dim?
lstm_cell_14/splitSplit%lstm_cell_14/split/split_dim:output:0lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_14/split?
lstm_cell_14/SigmoidSigmoidlstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid?
lstm_cell_14/Sigmoid_1Sigmoidlstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_1?
lstm_cell_14/mulMullstm_cell_14/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul}
lstm_cell_14/ReluRelulstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu?
lstm_cell_14/mul_1Mullstm_cell_14/Sigmoid:y:0lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_1?
lstm_cell_14/add_1AddV2lstm_cell_14/mul:z:0lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/add_1?
lstm_cell_14/Sigmoid_2Sigmoidlstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Sigmoid_2|
lstm_cell_14/Relu_1Relulstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/Relu_1?
lstm_cell_14/mul_2Mullstm_cell_14/Sigmoid_2:y:0!lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_14/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_14_matmul_readvariableop_resource-lstm_cell_14_matmul_1_readvariableop_resource,lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196145866*!
condR
while_cond_1196145865*K
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
IdentityIdentitytranspose_1:y:0$^lstm_cell_14/BiasAdd/ReadVariableOp#^lstm_cell_14/MatMul/ReadVariableOp%^lstm_cell_14/MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::2J
#lstm_cell_14/BiasAdd/ReadVariableOp#lstm_cell_14/BiasAdd/ReadVariableOp2H
"lstm_cell_14/MatMul/ReadVariableOp"lstm_cell_14/MatMul/ReadVariableOp2L
$lstm_cell_14/MatMul_1/ReadVariableOp$lstm_cell_14/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_price_layer1_layer_call_fn_1196147678
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
 :?????????????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961450222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :?????????????????? 2

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
G__inference_model_9_layer_call_and_return_conditional_losses_1196146569

inputs
inputs_1
price_layer1_1196146537
price_layer1_1196146539
price_layer1_1196146541
price_layer2_1196146544
price_layer2_1196146546
price_layer2_1196146548
fixed_layer1_1196146553
fixed_layer1_1196146555
fixed_layer2_1196146558
fixed_layer2_1196146560
action_output_1196146563
action_output_1196146565
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_1196146537price_layer1_1196146539price_layer1_1196146541*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961459512&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1196146544price_layer2_1196146546price_layer2_1196146548*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961462862&
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_11961463222
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_11961463372
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1196146553fixed_layer1_1196146555*
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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11961463572&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1196146558fixed_layer2_1196146560*
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11961463842&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1196146563action_output_1196146565*
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
M__inference_action_output_layer_call_and_return_conditional_losses_11961464102'
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
?%
?
while_body_1196144821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_lstm_cell_14_1196144845_0#
while_lstm_cell_14_1196144847_0#
while_lstm_cell_14_1196144849_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_lstm_cell_14_1196144845!
while_lstm_cell_14_1196144847!
while_lstm_cell_14_1196144849??*while/lstm_cell_14/StatefulPartitionedCall?
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
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_1196144845_0while_lstm_cell_14_1196144847_0while_lstm_cell_14_1196144849_0*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_11961444942,
*while/lstm_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_14_1196144845while_lstm_cell_14_1196144845_0"@
while_lstm_cell_14_1196144847while_lstm_cell_14_1196144847_0"@
while_lstm_cell_14_1196144849while_lstm_cell_14_1196144849_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 
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
?&
?
G__inference_model_9_layer_call_and_return_conditional_losses_1196146463
price_input
	env_input
price_layer1_1196146431
price_layer1_1196146433
price_layer1_1196146435
price_layer2_1196146438
price_layer2_1196146440
price_layer2_1196146442
fixed_layer1_1196146447
fixed_layer1_1196146449
fixed_layer2_1196146452
fixed_layer2_1196146454
action_output_1196146457
action_output_1196146459
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallprice_inputprice_layer1_1196146431price_layer1_1196146433price_layer1_1196146435*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961459512&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1196146438price_layer2_1196146440price_layer2_1196146442*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961462862&
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_11961463222
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_11961463372
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1196146447fixed_layer1_1196146449*
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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11961463572&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1196146452fixed_layer2_1196146454*
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11961463842&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1196146457action_output_1196146459*
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
M__inference_action_output_layer_call_and_return_conditional_losses_11961464102'
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
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196144494

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
:	?*
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
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
while_cond_1196148226
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196148226___redundant_placeholder08
4while_while_cond_1196148226___redundant_placeholder18
4while_while_cond_1196148226___redundant_placeholder28
4while_while_cond_1196148226___redundant_placeholder3
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
while_cond_1196147898
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196147898___redundant_placeholder08
4while_while_cond_1196147898___redundant_placeholder18
4while_while_cond_1196147898___redundant_placeholder28
4while_while_cond_1196147898___redundant_placeholder3
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
?
*model_9_price_layer2_while_cond_1196144311F
Bmodel_9_price_layer2_while_model_9_price_layer2_while_loop_counterL
Hmodel_9_price_layer2_while_model_9_price_layer2_while_maximum_iterations*
&model_9_price_layer2_while_placeholder,
(model_9_price_layer2_while_placeholder_1,
(model_9_price_layer2_while_placeholder_2,
(model_9_price_layer2_while_placeholder_3H
Dmodel_9_price_layer2_while_less_model_9_price_layer2_strided_slice_1b
^model_9_price_layer2_while_model_9_price_layer2_while_cond_1196144311___redundant_placeholder0b
^model_9_price_layer2_while_model_9_price_layer2_while_cond_1196144311___redundant_placeholder1b
^model_9_price_layer2_while_model_9_price_layer2_while_cond_1196144311___redundant_placeholder2b
^model_9_price_layer2_while_model_9_price_layer2_while_cond_1196144311___redundant_placeholder3'
#model_9_price_layer2_while_identity
?
model_9/price_layer2/while/LessLess&model_9_price_layer2_while_placeholderDmodel_9_price_layer2_while_less_model_9_price_layer2_strided_slice_1*
T0*
_output_shapes
: 2!
model_9/price_layer2/while/Less?
#model_9/price_layer2/while/IdentityIdentity#model_9/price_layer2/while/Less:z:0*
T0
*
_output_shapes
: 2%
#model_9/price_layer2/while/Identity"S
#model_9_price_layer2_while_identity,model_9/price_layer2/while/Identity:output:0*S
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
while_cond_1196145712
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196145712___redundant_placeholder08
4while_while_cond_1196145712___redundant_placeholder18
4while_while_cond_1196145712___redundant_placeholder28
4while_while_cond_1196145712___redundant_placeholder3
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
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196148778

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
:	?*
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
E:?????????:????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:?????????
 
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
?
x
L__inference_concat_layer_layer_call_and_return_conditional_losses_1196148680
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
?
v
L__inference_concat_layer_layer_call_and_return_conditional_losses_1196146337

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
?
?
while_cond_1196147417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196147417___redundant_placeholder08
4while_while_cond_1196147417___redundant_placeholder18
4while_while_cond_1196147417___redundant_placeholder28
4while_while_cond_1196147417___redundant_placeholder3
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
?
?
1__inference_fixed_layer1_layer_call_fn_1196148706

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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11961463572
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
?E
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196145500

inputs
lstm_cell_15_1196145418
lstm_cell_15_1196145420
lstm_cell_15_1196145422
identity??$lstm_cell_15/StatefulPartitionedCall?whileD
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
 :?????????????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_1196145418lstm_cell_15_1196145420lstm_cell_15_1196145422*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_11961451042&
$lstm_cell_15/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_1196145418lstm_cell_15_1196145420lstm_cell_15_1196145422*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196145431*!
condR
while_cond_1196145430*K
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
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_15/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????????????? :::2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?
?
"price_layer2_while_cond_11961471806
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_38
4price_layer2_while_less_price_layer2_strided_slice_1R
Nprice_layer2_while_price_layer2_while_cond_1196147180___redundant_placeholder0R
Nprice_layer2_while_price_layer2_while_cond_1196147180___redundant_placeholder1R
Nprice_layer2_while_price_layer2_while_cond_1196147180___redundant_placeholder2R
Nprice_layer2_while_price_layer2_while_cond_1196147180___redundant_placeholder3
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
?
]
1__inference_concat_layer_layer_call_fn_1196148686
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_11961463372
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

?
,__inference_model_9_layer_call_fn_1196146596
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11961465692
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
?
?
1__inference_fixed_layer2_layer_call_fn_1196148726

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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11961463842
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
?C
?
while_body_1196147571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
while_cond_1196144952
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196144952___redundant_placeholder08
4while_while_cond_1196144952___redundant_placeholder18
4while_while_cond_1196144952___redundant_placeholder28
4while_while_cond_1196144952___redundant_placeholder3
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
?[
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196146286

inputs/
+lstm_cell_15_matmul_readvariableop_resource1
-lstm_cell_15_matmul_1_readvariableop_resource0
,lstm_cell_15_biasadd_readvariableop_resource
identity??#lstm_cell_15/BiasAdd/ReadVariableOp?"lstm_cell_15/MatMul/ReadVariableOp?$lstm_cell_15/MatMul_1/ReadVariableOp?whileD
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
:????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
"lstm_cell_15/MatMul/ReadVariableOpReadVariableOp+lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype02$
"lstm_cell_15/MatMul/ReadVariableOp?
lstm_cell_15/MatMulMatMulstrided_slice_2:output:0*lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul?
$lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype02&
$lstm_cell_15/MatMul_1/ReadVariableOp?
lstm_cell_15/MatMul_1MatMulzeros:output:0,lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/MatMul_1?
lstm_cell_15/addAddV2lstm_cell_15/MatMul:product:0lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/add?
#lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#lstm_cell_15/BiasAdd/ReadVariableOp?
lstm_cell_15/BiasAddBiasAddlstm_cell_15/add:z:0+lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
lstm_cell_15/BiasAddj
lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/Const~
lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_15/split/split_dim?
lstm_cell_15/splitSplit%lstm_cell_15/split/split_dim:output:0lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
lstm_cell_15/split?
lstm_cell_15/SigmoidSigmoidlstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid?
lstm_cell_15/Sigmoid_1Sigmoidlstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_1?
lstm_cell_15/mulMullstm_cell_15/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul}
lstm_cell_15/ReluRelulstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu?
lstm_cell_15/mul_1Mullstm_cell_15/Sigmoid:y:0lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_1?
lstm_cell_15/add_1AddV2lstm_cell_15/mul:z:0lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/add_1?
lstm_cell_15/Sigmoid_2Sigmoidlstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Sigmoid_2|
lstm_cell_15/Relu_1Relulstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/Relu_1?
lstm_cell_15/mul_2Mullstm_cell_15/Sigmoid_2:y:0!lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
lstm_cell_15/mul_2?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_15_matmul_readvariableop_resource-lstm_cell_15_matmul_1_readvariableop_resource,lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196146201*!
condR
while_cond_1196146200*K
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
IdentityIdentitystrided_slice_3:output:0$^lstm_cell_15/BiasAdd/ReadVariableOp#^lstm_cell_15/MatMul/ReadVariableOp%^lstm_cell_15/MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? :::2J
#lstm_cell_15/BiasAdd/ReadVariableOp#lstm_cell_15/BiasAdd/ReadVariableOp2H
"lstm_cell_15/MatMul/ReadVariableOp"lstm_cell_15/MatMul/ReadVariableOp2L
$lstm_cell_15/MatMul_1/ReadVariableOp$lstm_cell_15/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
"price_layer1_while_cond_11961467046
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_38
4price_layer1_while_less_price_layer1_strided_slice_1R
Nprice_layer1_while_price_layer1_while_cond_1196146704___redundant_placeholder0R
Nprice_layer1_while_price_layer1_while_cond_1196146704___redundant_placeholder1R
Nprice_layer1_while_price_layer1_while_cond_1196146704___redundant_placeholder2R
Nprice_layer1_while_price_layer1_while_cond_1196146704___redundant_placeholder3
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
?b
?
*model_9_price_layer2_while_body_1196144312F
Bmodel_9_price_layer2_while_model_9_price_layer2_while_loop_counterL
Hmodel_9_price_layer2_while_model_9_price_layer2_while_maximum_iterations*
&model_9_price_layer2_while_placeholder,
(model_9_price_layer2_while_placeholder_1,
(model_9_price_layer2_while_placeholder_2,
(model_9_price_layer2_while_placeholder_3E
Amodel_9_price_layer2_while_model_9_price_layer2_strided_slice_1_0?
}model_9_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer2_tensorarrayunstack_tensorlistfromtensor_0L
Hmodel_9_price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0N
Jmodel_9_price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0M
Imodel_9_price_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0'
#model_9_price_layer2_while_identity)
%model_9_price_layer2_while_identity_1)
%model_9_price_layer2_while_identity_2)
%model_9_price_layer2_while_identity_3)
%model_9_price_layer2_while_identity_4)
%model_9_price_layer2_while_identity_5C
?model_9_price_layer2_while_model_9_price_layer2_strided_slice_1
{model_9_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer2_tensorarrayunstack_tensorlistfromtensorJ
Fmodel_9_price_layer2_while_lstm_cell_15_matmul_readvariableop_resourceL
Hmodel_9_price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resourceK
Gmodel_9_price_layer2_while_lstm_cell_15_biasadd_readvariableop_resource??>model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp?=model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp??model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?
Lmodel_9/price_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2N
Lmodel_9/price_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>model_9/price_layer2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_9_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer2_tensorarrayunstack_tensorlistfromtensor_0&model_9_price_layer2_while_placeholderUmodel_9/price_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02@
>model_9/price_layer2/while/TensorArrayV2Read/TensorListGetItem?
=model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOpHmodel_9_price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02?
=model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp?
.model_9/price_layer2/while/lstm_cell_15/MatMulMatMulEmodel_9/price_layer2/while/TensorArrayV2Read/TensorListGetItem:item:0Emodel_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.model_9/price_layer2/while/lstm_cell_15/MatMul?
?model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpJmodel_9_price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02A
?model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?
0model_9/price_layer2/while/lstm_cell_15/MatMul_1MatMul(model_9_price_layer2_while_placeholder_2Gmodel_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model_9/price_layer2/while/lstm_cell_15/MatMul_1?
+model_9/price_layer2/while/lstm_cell_15/addAddV28model_9/price_layer2/while/lstm_cell_15/MatMul:product:0:model_9/price_layer2/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+model_9/price_layer2/while/lstm_cell_15/add?
>model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpImodel_9_price_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp?
/model_9/price_layer2/while/lstm_cell_15/BiasAddBiasAdd/model_9/price_layer2/while/lstm_cell_15/add:z:0Fmodel_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/model_9/price_layer2/while/lstm_cell_15/BiasAdd?
-model_9/price_layer2/while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2/
-model_9/price_layer2/while/lstm_cell_15/Const?
7model_9/price_layer2/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7model_9/price_layer2/while/lstm_cell_15/split/split_dim?
-model_9/price_layer2/while/lstm_cell_15/splitSplit@model_9/price_layer2/while/lstm_cell_15/split/split_dim:output:08model_9/price_layer2/while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2/
-model_9/price_layer2/while/lstm_cell_15/split?
/model_9/price_layer2/while/lstm_cell_15/SigmoidSigmoid6model_9/price_layer2/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 21
/model_9/price_layer2/while/lstm_cell_15/Sigmoid?
1model_9/price_layer2/while/lstm_cell_15/Sigmoid_1Sigmoid6model_9/price_layer2/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 23
1model_9/price_layer2/while/lstm_cell_15/Sigmoid_1?
+model_9/price_layer2/while/lstm_cell_15/mulMul5model_9/price_layer2/while/lstm_cell_15/Sigmoid_1:y:0(model_9_price_layer2_while_placeholder_3*
T0*'
_output_shapes
:????????? 2-
+model_9/price_layer2/while/lstm_cell_15/mul?
,model_9/price_layer2/while/lstm_cell_15/ReluRelu6model_9/price_layer2/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2.
,model_9/price_layer2/while/lstm_cell_15/Relu?
-model_9/price_layer2/while/lstm_cell_15/mul_1Mul3model_9/price_layer2/while/lstm_cell_15/Sigmoid:y:0:model_9/price_layer2/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2/
-model_9/price_layer2/while/lstm_cell_15/mul_1?
-model_9/price_layer2/while/lstm_cell_15/add_1AddV2/model_9/price_layer2/while/lstm_cell_15/mul:z:01model_9/price_layer2/while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2/
-model_9/price_layer2/while/lstm_cell_15/add_1?
1model_9/price_layer2/while/lstm_cell_15/Sigmoid_2Sigmoid6model_9/price_layer2/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 23
1model_9/price_layer2/while/lstm_cell_15/Sigmoid_2?
.model_9/price_layer2/while/lstm_cell_15/Relu_1Relu1model_9/price_layer2/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 20
.model_9/price_layer2/while/lstm_cell_15/Relu_1?
-model_9/price_layer2/while/lstm_cell_15/mul_2Mul5model_9/price_layer2/while/lstm_cell_15/Sigmoid_2:y:0<model_9/price_layer2/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2/
-model_9/price_layer2/while/lstm_cell_15/mul_2?
?model_9/price_layer2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_9_price_layer2_while_placeholder_1&model_9_price_layer2_while_placeholder1model_9/price_layer2/while/lstm_cell_15/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?model_9/price_layer2/while/TensorArrayV2Write/TensorListSetItem?
 model_9/price_layer2/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_9/price_layer2/while/add/y?
model_9/price_layer2/while/addAddV2&model_9_price_layer2_while_placeholder)model_9/price_layer2/while/add/y:output:0*
T0*
_output_shapes
: 2 
model_9/price_layer2/while/add?
"model_9/price_layer2/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/price_layer2/while/add_1/y?
 model_9/price_layer2/while/add_1AddV2Bmodel_9_price_layer2_while_model_9_price_layer2_while_loop_counter+model_9/price_layer2/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 model_9/price_layer2/while/add_1?
#model_9/price_layer2/while/IdentityIdentity$model_9/price_layer2/while/add_1:z:0?^model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>^model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp@^model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model_9/price_layer2/while/Identity?
%model_9/price_layer2/while/Identity_1IdentityHmodel_9_price_layer2_while_model_9_price_layer2_while_maximum_iterations?^model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>^model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp@^model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_9/price_layer2/while/Identity_1?
%model_9/price_layer2/while/Identity_2Identity"model_9/price_layer2/while/add:z:0?^model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>^model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp@^model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_9/price_layer2/while/Identity_2?
%model_9/price_layer2/while/Identity_3IdentityOmodel_9/price_layer2/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>^model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp@^model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_9/price_layer2/while/Identity_3?
%model_9/price_layer2/while/Identity_4Identity1model_9/price_layer2/while/lstm_cell_15/mul_2:z:0?^model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>^model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp@^model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2'
%model_9/price_layer2/while/Identity_4?
%model_9/price_layer2/while/Identity_5Identity1model_9/price_layer2/while/lstm_cell_15/add_1:z:0?^model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>^model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp@^model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2'
%model_9/price_layer2/while/Identity_5"S
#model_9_price_layer2_while_identity,model_9/price_layer2/while/Identity:output:0"W
%model_9_price_layer2_while_identity_1.model_9/price_layer2/while/Identity_1:output:0"W
%model_9_price_layer2_while_identity_2.model_9/price_layer2/while/Identity_2:output:0"W
%model_9_price_layer2_while_identity_3.model_9/price_layer2/while/Identity_3:output:0"W
%model_9_price_layer2_while_identity_4.model_9/price_layer2/while/Identity_4:output:0"W
%model_9_price_layer2_while_identity_5.model_9/price_layer2/while/Identity_5:output:0"?
Gmodel_9_price_layer2_while_lstm_cell_15_biasadd_readvariableop_resourceImodel_9_price_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0"?
Hmodel_9_price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resourceJmodel_9_price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0"?
Fmodel_9_price_layer2_while_lstm_cell_15_matmul_readvariableop_resourceHmodel_9_price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0"?
?model_9_price_layer2_while_model_9_price_layer2_strided_slice_1Amodel_9_price_layer2_while_model_9_price_layer2_strided_slice_1_0"?
{model_9_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer2_tensorarrayunstack_tensorlistfromtensor}model_9_price_layer2_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2?
>model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp>model_9/price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp2~
=model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp=model_9/price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp2?
?model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?model_9/price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
?C
?
while_body_1196148227
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_15_matmul_readvariableop_resource_09
5while_lstm_cell_15_matmul_1_readvariableop_resource_08
4while_lstm_cell_15_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_15_matmul_readvariableop_resource7
3while_lstm_cell_15_matmul_1_readvariableop_resource6
2while_lstm_cell_15_biasadd_readvariableop_resource??)while/lstm_cell_15/BiasAdd/ReadVariableOp?(while/lstm_cell_15/MatMul/ReadVariableOp?*while/lstm_cell_15/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp?
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul?
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp?
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul_1?
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/add?
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp?
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/BiasAddv
while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_15/Const?
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim?
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_15/split?
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid?
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_1?
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul?
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu?
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_1?
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/add_1?
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_2?
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu_1?
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
?
?
1__inference_price_layer2_layer_call_fn_1196148662
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961456322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?
?
while_cond_1196144820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196144820___redundant_placeholder08
4while_while_cond_1196144820___redundant_placeholder18
4while_while_cond_1196144820___redundant_placeholder28
4while_while_cond_1196144820___redundant_placeholder3
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
?C
?
while_body_1196148074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_15_matmul_readvariableop_resource_09
5while_lstm_cell_15_matmul_1_readvariableop_resource_08
4while_lstm_cell_15_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_15_matmul_readvariableop_resource7
3while_lstm_cell_15_matmul_1_readvariableop_resource6
2while_lstm_cell_15_biasadd_readvariableop_resource??)while/lstm_cell_15/BiasAdd/ReadVariableOp?(while/lstm_cell_15/MatMul/ReadVariableOp?*while/lstm_cell_15/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp?
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul?
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp?
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul_1?
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/add?
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp?
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/BiasAddv
while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_15/Const?
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim?
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_15/split?
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid?
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_1?
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul?
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu?
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_1?
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/add_1?
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_2?
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu_1?
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
?b
?
*model_9_price_layer1_while_body_1196144163F
Bmodel_9_price_layer1_while_model_9_price_layer1_while_loop_counterL
Hmodel_9_price_layer1_while_model_9_price_layer1_while_maximum_iterations*
&model_9_price_layer1_while_placeholder,
(model_9_price_layer1_while_placeholder_1,
(model_9_price_layer1_while_placeholder_2,
(model_9_price_layer1_while_placeholder_3E
Amodel_9_price_layer1_while_model_9_price_layer1_strided_slice_1_0?
}model_9_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer1_tensorarrayunstack_tensorlistfromtensor_0L
Hmodel_9_price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0N
Jmodel_9_price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0M
Imodel_9_price_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0'
#model_9_price_layer1_while_identity)
%model_9_price_layer1_while_identity_1)
%model_9_price_layer1_while_identity_2)
%model_9_price_layer1_while_identity_3)
%model_9_price_layer1_while_identity_4)
%model_9_price_layer1_while_identity_5C
?model_9_price_layer1_while_model_9_price_layer1_strided_slice_1
{model_9_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer1_tensorarrayunstack_tensorlistfromtensorJ
Fmodel_9_price_layer1_while_lstm_cell_14_matmul_readvariableop_resourceL
Hmodel_9_price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resourceK
Gmodel_9_price_layer1_while_lstm_cell_14_biasadd_readvariableop_resource??>model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp?=model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp??model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?
Lmodel_9/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2N
Lmodel_9/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape?
>model_9/price_layer1/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}model_9_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer1_tensorarrayunstack_tensorlistfromtensor_0&model_9_price_layer1_while_placeholderUmodel_9/price_layer1/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:?????????*
element_dtype02@
>model_9/price_layer1/while/TensorArrayV2Read/TensorListGetItem?
=model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOpHmodel_9_price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02?
=model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp?
.model_9/price_layer1/while/lstm_cell_14/MatMulMatMulEmodel_9/price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0Emodel_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????20
.model_9/price_layer1/while/lstm_cell_14/MatMul?
?model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpJmodel_9_price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02A
?model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?
0model_9/price_layer1/while/lstm_cell_14/MatMul_1MatMul(model_9_price_layer1_while_placeholder_2Gmodel_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????22
0model_9/price_layer1/while/lstm_cell_14/MatMul_1?
+model_9/price_layer1/while/lstm_cell_14/addAddV28model_9/price_layer1/while/lstm_cell_14/MatMul:product:0:model_9/price_layer1/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2-
+model_9/price_layer1/while/lstm_cell_14/add?
>model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpImodel_9_price_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02@
>model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp?
/model_9/price_layer1/while/lstm_cell_14/BiasAddBiasAdd/model_9/price_layer1/while/lstm_cell_14/add:z:0Fmodel_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????21
/model_9/price_layer1/while/lstm_cell_14/BiasAdd?
-model_9/price_layer1/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2/
-model_9/price_layer1/while/lstm_cell_14/Const?
7model_9/price_layer1/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7model_9/price_layer1/while/lstm_cell_14/split/split_dim?
-model_9/price_layer1/while/lstm_cell_14/splitSplit@model_9/price_layer1/while/lstm_cell_14/split/split_dim:output:08model_9/price_layer1/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2/
-model_9/price_layer1/while/lstm_cell_14/split?
/model_9/price_layer1/while/lstm_cell_14/SigmoidSigmoid6model_9/price_layer1/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 21
/model_9/price_layer1/while/lstm_cell_14/Sigmoid?
1model_9/price_layer1/while/lstm_cell_14/Sigmoid_1Sigmoid6model_9/price_layer1/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 23
1model_9/price_layer1/while/lstm_cell_14/Sigmoid_1?
+model_9/price_layer1/while/lstm_cell_14/mulMul5model_9/price_layer1/while/lstm_cell_14/Sigmoid_1:y:0(model_9_price_layer1_while_placeholder_3*
T0*'
_output_shapes
:????????? 2-
+model_9/price_layer1/while/lstm_cell_14/mul?
,model_9/price_layer1/while/lstm_cell_14/ReluRelu6model_9/price_layer1/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2.
,model_9/price_layer1/while/lstm_cell_14/Relu?
-model_9/price_layer1/while/lstm_cell_14/mul_1Mul3model_9/price_layer1/while/lstm_cell_14/Sigmoid:y:0:model_9/price_layer1/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2/
-model_9/price_layer1/while/lstm_cell_14/mul_1?
-model_9/price_layer1/while/lstm_cell_14/add_1AddV2/model_9/price_layer1/while/lstm_cell_14/mul:z:01model_9/price_layer1/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2/
-model_9/price_layer1/while/lstm_cell_14/add_1?
1model_9/price_layer1/while/lstm_cell_14/Sigmoid_2Sigmoid6model_9/price_layer1/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 23
1model_9/price_layer1/while/lstm_cell_14/Sigmoid_2?
.model_9/price_layer1/while/lstm_cell_14/Relu_1Relu1model_9/price_layer1/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 20
.model_9/price_layer1/while/lstm_cell_14/Relu_1?
-model_9/price_layer1/while/lstm_cell_14/mul_2Mul5model_9/price_layer1/while/lstm_cell_14/Sigmoid_2:y:0<model_9/price_layer1/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2/
-model_9/price_layer1/while/lstm_cell_14/mul_2?
?model_9/price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(model_9_price_layer1_while_placeholder_1&model_9_price_layer1_while_placeholder1model_9/price_layer1/while/lstm_cell_14/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?model_9/price_layer1/while/TensorArrayV2Write/TensorListSetItem?
 model_9/price_layer1/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 model_9/price_layer1/while/add/y?
model_9/price_layer1/while/addAddV2&model_9_price_layer1_while_placeholder)model_9/price_layer1/while/add/y:output:0*
T0*
_output_shapes
: 2 
model_9/price_layer1/while/add?
"model_9/price_layer1/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"model_9/price_layer1/while/add_1/y?
 model_9/price_layer1/while/add_1AddV2Bmodel_9_price_layer1_while_model_9_price_layer1_while_loop_counter+model_9/price_layer1/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 model_9/price_layer1/while/add_1?
#model_9/price_layer1/while/IdentityIdentity$model_9/price_layer1/while/add_1:z:0?^model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>^model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp@^model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2%
#model_9/price_layer1/while/Identity?
%model_9/price_layer1/while/Identity_1IdentityHmodel_9_price_layer1_while_model_9_price_layer1_while_maximum_iterations?^model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>^model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp@^model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_9/price_layer1/while/Identity_1?
%model_9/price_layer1/while/Identity_2Identity"model_9/price_layer1/while/add:z:0?^model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>^model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp@^model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_9/price_layer1/while/Identity_2?
%model_9/price_layer1/while/Identity_3IdentityOmodel_9/price_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:0?^model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>^model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp@^model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_9/price_layer1/while/Identity_3?
%model_9/price_layer1/while/Identity_4Identity1model_9/price_layer1/while/lstm_cell_14/mul_2:z:0?^model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>^model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp@^model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2'
%model_9/price_layer1/while/Identity_4?
%model_9/price_layer1/while/Identity_5Identity1model_9/price_layer1/while/lstm_cell_14/add_1:z:0?^model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>^model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp@^model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2'
%model_9/price_layer1/while/Identity_5"S
#model_9_price_layer1_while_identity,model_9/price_layer1/while/Identity:output:0"W
%model_9_price_layer1_while_identity_1.model_9/price_layer1/while/Identity_1:output:0"W
%model_9_price_layer1_while_identity_2.model_9/price_layer1/while/Identity_2:output:0"W
%model_9_price_layer1_while_identity_3.model_9/price_layer1/while/Identity_3:output:0"W
%model_9_price_layer1_while_identity_4.model_9/price_layer1/while/Identity_4:output:0"W
%model_9_price_layer1_while_identity_5.model_9/price_layer1/while/Identity_5:output:0"?
Gmodel_9_price_layer1_while_lstm_cell_14_biasadd_readvariableop_resourceImodel_9_price_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0"?
Hmodel_9_price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resourceJmodel_9_price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0"?
Fmodel_9_price_layer1_while_lstm_cell_14_matmul_readvariableop_resourceHmodel_9_price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0"?
?model_9_price_layer1_while_model_9_price_layer1_strided_slice_1Amodel_9_price_layer1_while_model_9_price_layer1_strided_slice_1_0"?
{model_9_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer1_tensorarrayunstack_tensorlistfromtensor}model_9_price_layer1_while_tensorarrayv2read_tensorlistgetitem_model_9_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2?
>model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp>model_9/price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp2~
=model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp=model_9/price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp2?
?model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?model_9/price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
1__inference_lstm_cell_15_layer_call_fn_1196148928

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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_11961451042
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
E:????????? :????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
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
?
?
while_cond_1196148554
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196148554___redundant_placeholder08
4while_while_cond_1196148554___redundant_placeholder18
4while_while_cond_1196148554___redundant_placeholder28
4while_while_cond_1196148554___redundant_placeholder3
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
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196148878

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
:	 ?*
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
E:????????? :????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
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
?
N
2__inference_price_flatten_layer_call_fn_1196148673

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
M__inference_price_flatten_layer_call_and_return_conditional_losses_11961463222
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

?
(__inference_signature_wrapper_1196146636
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
GPU 2J 8? *.
f)R'
%__inference__wrapped_model_11961444212
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
?
?
*model_9_price_layer1_while_cond_1196144162F
Bmodel_9_price_layer1_while_model_9_price_layer1_while_loop_counterL
Hmodel_9_price_layer1_while_model_9_price_layer1_while_maximum_iterations*
&model_9_price_layer1_while_placeholder,
(model_9_price_layer1_while_placeholder_1,
(model_9_price_layer1_while_placeholder_2,
(model_9_price_layer1_while_placeholder_3H
Dmodel_9_price_layer1_while_less_model_9_price_layer1_strided_slice_1b
^model_9_price_layer1_while_model_9_price_layer1_while_cond_1196144162___redundant_placeholder0b
^model_9_price_layer1_while_model_9_price_layer1_while_cond_1196144162___redundant_placeholder1b
^model_9_price_layer1_while_model_9_price_layer1_while_cond_1196144162___redundant_placeholder2b
^model_9_price_layer1_while_model_9_price_layer1_while_cond_1196144162___redundant_placeholder3'
#model_9_price_layer1_while_identity
?
model_9/price_layer1/while/LessLess&model_9_price_layer1_while_placeholderDmodel_9_price_layer1_while_less_model_9_price_layer1_strided_slice_1*
T0*
_output_shapes
: 2!
model_9/price_layer1/while/Less?
#model_9/price_layer1/while/IdentityIdentity#model_9/price_layer1/while/Less:z:0*
T0
*
_output_shapes
: 2%
#model_9/price_layer1/while/Identity"S
#model_9_price_layer1_while_identity,model_9/price_layer1/while/Identity:output:0*S
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
while_cond_1196145430
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196145430___redundant_placeholder08
4while_while_cond_1196145430___redundant_placeholder18
4while_while_cond_1196145430___redundant_placeholder28
4while_while_cond_1196145430___redundant_placeholder3
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
?V
?
"price_layer2_while_body_11961471816
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_35
1price_layer2_while_price_layer2_strided_slice_1_0q
mprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0D
@price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0F
Bprice_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0E
Aprice_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0
price_layer2_while_identity!
price_layer2_while_identity_1!
price_layer2_while_identity_2!
price_layer2_while_identity_3!
price_layer2_while_identity_4!
price_layer2_while_identity_53
/price_layer2_while_price_layer2_strided_slice_1o
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensorB
>price_layer2_while_lstm_cell_15_matmul_readvariableop_resourceD
@price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resourceC
?price_layer2_while_lstm_cell_15_biasadd_readvariableop_resource??6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp?5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp?7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2F
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6price_layer2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0price_layer2_while_placeholderMprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype028
6price_layer2/while/TensorArrayV2Read/TensorListGetItem?
5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp@price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype027
5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp?
&price_layer2/while/lstm_cell_15/MatMulMatMul=price_layer2/while/TensorArrayV2Read/TensorListGetItem:item:0=price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&price_layer2/while/lstm_cell_15/MatMul?
7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpBprice_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?
(price_layer2/while/lstm_cell_15/MatMul_1MatMul price_layer2_while_placeholder_2?price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(price_layer2/while/lstm_cell_15/MatMul_1?
#price_layer2/while/lstm_cell_15/addAddV20price_layer2/while/lstm_cell_15/MatMul:product:02price_layer2/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#price_layer2/while/lstm_cell_15/add?
6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpAprice_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype028
6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp?
'price_layer2/while/lstm_cell_15/BiasAddBiasAdd'price_layer2/while/lstm_cell_15/add:z:0>price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'price_layer2/while/lstm_cell_15/BiasAdd?
%price_layer2/while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%price_layer2/while/lstm_cell_15/Const?
/price_layer2/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/price_layer2/while/lstm_cell_15/split/split_dim?
%price_layer2/while/lstm_cell_15/splitSplit8price_layer2/while/lstm_cell_15/split/split_dim:output:00price_layer2/while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2'
%price_layer2/while/lstm_cell_15/split?
'price_layer2/while/lstm_cell_15/SigmoidSigmoid.price_layer2/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2)
'price_layer2/while/lstm_cell_15/Sigmoid?
)price_layer2/while/lstm_cell_15/Sigmoid_1Sigmoid.price_layer2/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2+
)price_layer2/while/lstm_cell_15/Sigmoid_1?
#price_layer2/while/lstm_cell_15/mulMul-price_layer2/while/lstm_cell_15/Sigmoid_1:y:0 price_layer2_while_placeholder_3*
T0*'
_output_shapes
:????????? 2%
#price_layer2/while/lstm_cell_15/mul?
$price_layer2/while/lstm_cell_15/ReluRelu.price_layer2/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_15/Relu?
%price_layer2/while/lstm_cell_15/mul_1Mul+price_layer2/while/lstm_cell_15/Sigmoid:y:02price_layer2/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_15/mul_1?
%price_layer2/while/lstm_cell_15/add_1AddV2'price_layer2/while/lstm_cell_15/mul:z:0)price_layer2/while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_15/add_1?
)price_layer2/while/lstm_cell_15/Sigmoid_2Sigmoid.price_layer2/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2+
)price_layer2/while/lstm_cell_15/Sigmoid_2?
&price_layer2/while/lstm_cell_15/Relu_1Relu)price_layer2/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2(
&price_layer2/while/lstm_cell_15/Relu_1?
%price_layer2/while/lstm_cell_15/mul_2Mul-price_layer2/while/lstm_cell_15/Sigmoid_2:y:04price_layer2/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_15/mul_2?
7price_layer2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer2_while_placeholder_1price_layer2_while_placeholder)price_layer2/while/lstm_cell_15/mul_2:z:0*
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
price_layer2/while/IdentityIdentityprice_layer2/while/add_1:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity?
price_layer2/while/Identity_1Identity8price_layer2_while_price_layer2_while_maximum_iterations7^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_1?
price_layer2/while/Identity_2Identityprice_layer2/while/add:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_2?
price_layer2/while/Identity_3IdentityGprice_layer2/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_3?
price_layer2/while/Identity_4Identity)price_layer2/while/lstm_cell_15/mul_2:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer2/while/Identity_4?
price_layer2/while/Identity_5Identity)price_layer2/while/lstm_cell_15/add_1:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
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
?price_layer2_while_lstm_cell_15_biasadd_readvariableop_resourceAprice_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0"?
@price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resourceBprice_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0"?
>price_layer2_while_lstm_cell_15_matmul_readvariableop_resource@price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0"d
/price_layer2_while_price_layer2_strided_slice_11price_layer2_while_price_layer2_strided_slice_1_0"?
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensormprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2p
6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp2n
5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp2r
7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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

?
,__inference_model_9_layer_call_fn_1196147320
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
GPU 2J 8? *P
fKRI
G__inference_model_9_layer_call_and_return_conditional_losses_11961465032
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
?
?
1__inference_price_layer2_layer_call_fn_1196148651
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961455002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0
?C
?
while_body_1196146201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_15_matmul_readvariableop_resource_09
5while_lstm_cell_15_matmul_1_readvariableop_resource_08
4while_lstm_cell_15_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_15_matmul_readvariableop_resource7
3while_lstm_cell_15_matmul_1_readvariableop_resource6
2while_lstm_cell_15_biasadd_readvariableop_resource??)while/lstm_cell_15/BiasAdd/ReadVariableOp?(while/lstm_cell_15/MatMul/ReadVariableOp?*while/lstm_cell_15/MatMul_1/ReadVariableOp?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
(while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02*
(while/lstm_cell_15/MatMul/ReadVariableOp?
while/lstm_cell_15/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul?
*while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_15/MatMul_1/ReadVariableOp?
while/lstm_cell_15/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/MatMul_1?
while/lstm_cell_15/addAddV2#while/lstm_cell_15/MatMul:product:0%while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/add?
)while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_15/BiasAdd/ReadVariableOp?
while/lstm_cell_15/BiasAddBiasAddwhile/lstm_cell_15/add:z:01while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_15/BiasAddv
while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_15/Const?
"while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_15/split/split_dim?
while/lstm_cell_15/splitSplit+while/lstm_cell_15/split/split_dim:output:0#while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_15/split?
while/lstm_cell_15/SigmoidSigmoid!while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid?
while/lstm_cell_15/Sigmoid_1Sigmoid!while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_1?
while/lstm_cell_15/mulMul while/lstm_cell_15/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul?
while/lstm_cell_15/ReluRelu!while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu?
while/lstm_cell_15/mul_1Mulwhile/lstm_cell_15/Sigmoid:y:0%while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_1?
while/lstm_cell_15/add_1AddV2while/lstm_cell_15/mul:z:0while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/add_1?
while/lstm_cell_15/Sigmoid_2Sigmoid!while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Sigmoid_2?
while/lstm_cell_15/Relu_1Reluwhile/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/Relu_1?
while/lstm_cell_15/mul_2Mul while/lstm_cell_15/Sigmoid_2:y:0'while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_15/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_15/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_15/mul_2:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_15/add_1:z:0*^while/lstm_cell_15/BiasAdd/ReadVariableOp)^while/lstm_cell_15/MatMul/ReadVariableOp+^while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_15_biasadd_readvariableop_resource4while_lstm_cell_15_biasadd_readvariableop_resource_0"l
3while_lstm_cell_15_matmul_1_readvariableop_resource5while_lstm_cell_15_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_15_matmul_readvariableop_resource3while_lstm_cell_15_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_15/BiasAdd/ReadVariableOp)while/lstm_cell_15/BiasAdd/ReadVariableOp2T
(while/lstm_cell_15/MatMul/ReadVariableOp(while/lstm_cell_15/MatMul/ReadVariableOp2X
*while/lstm_cell_15/MatMul_1/ReadVariableOp*while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
?V
?
"price_layer1_while_body_11961467056
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_35
1price_layer1_while_price_layer1_strided_slice_1_0q
mprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0D
@price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0F
Bprice_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0E
Aprice_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0
price_layer1_while_identity!
price_layer1_while_identity_1!
price_layer1_while_identity_2!
price_layer1_while_identity_3!
price_layer1_while_identity_4!
price_layer1_while_identity_53
/price_layer1_while_price_layer1_strided_slice_1o
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensorB
>price_layer1_while_lstm_cell_14_matmul_readvariableop_resourceD
@price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resourceC
?price_layer1_while_lstm_cell_14_biasadd_readvariableop_resource??6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp?5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp?7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp@price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype027
5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp?
&price_layer1/while/lstm_cell_14/MatMulMatMul=price_layer1/while/TensorArrayV2Read/TensorListGetItem:item:0=price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&price_layer1/while/lstm_cell_14/MatMul?
7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOpBprice_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp?
(price_layer1/while/lstm_cell_14/MatMul_1MatMul price_layer1_while_placeholder_2?price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(price_layer1/while/lstm_cell_14/MatMul_1?
#price_layer1/while/lstm_cell_14/addAddV20price_layer1/while/lstm_cell_14/MatMul:product:02price_layer1/while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#price_layer1/while/lstm_cell_14/add?
6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOpAprice_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype028
6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp?
'price_layer1/while/lstm_cell_14/BiasAddBiasAdd'price_layer1/while/lstm_cell_14/add:z:0>price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'price_layer1/while/lstm_cell_14/BiasAdd?
%price_layer1/while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%price_layer1/while/lstm_cell_14/Const?
/price_layer1/while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/price_layer1/while/lstm_cell_14/split/split_dim?
%price_layer1/while/lstm_cell_14/splitSplit8price_layer1/while/lstm_cell_14/split/split_dim:output:00price_layer1/while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2'
%price_layer1/while/lstm_cell_14/split?
'price_layer1/while/lstm_cell_14/SigmoidSigmoid.price_layer1/while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2)
'price_layer1/while/lstm_cell_14/Sigmoid?
)price_layer1/while/lstm_cell_14/Sigmoid_1Sigmoid.price_layer1/while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2+
)price_layer1/while/lstm_cell_14/Sigmoid_1?
#price_layer1/while/lstm_cell_14/mulMul-price_layer1/while/lstm_cell_14/Sigmoid_1:y:0 price_layer1_while_placeholder_3*
T0*'
_output_shapes
:????????? 2%
#price_layer1/while/lstm_cell_14/mul?
$price_layer1/while/lstm_cell_14/ReluRelu.price_layer1/while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2&
$price_layer1/while/lstm_cell_14/Relu?
%price_layer1/while/lstm_cell_14/mul_1Mul+price_layer1/while/lstm_cell_14/Sigmoid:y:02price_layer1/while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_14/mul_1?
%price_layer1/while/lstm_cell_14/add_1AddV2'price_layer1/while/lstm_cell_14/mul:z:0)price_layer1/while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_14/add_1?
)price_layer1/while/lstm_cell_14/Sigmoid_2Sigmoid.price_layer1/while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2+
)price_layer1/while/lstm_cell_14/Sigmoid_2?
&price_layer1/while/lstm_cell_14/Relu_1Relu)price_layer1/while/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2(
&price_layer1/while/lstm_cell_14/Relu_1?
%price_layer1/while/lstm_cell_14/mul_2Mul-price_layer1/while/lstm_cell_14/Sigmoid_2:y:04price_layer1/while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer1/while/lstm_cell_14/mul_2?
7price_layer1/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer1_while_placeholder_1price_layer1_while_placeholder)price_layer1/while/lstm_cell_14/mul_2:z:0*
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
price_layer1/while/IdentityIdentityprice_layer1/while/add_1:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity?
price_layer1/while/Identity_1Identity8price_layer1_while_price_layer1_while_maximum_iterations7^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_1?
price_layer1/while/Identity_2Identityprice_layer1/while/add:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_2?
price_layer1/while/Identity_3IdentityGprice_layer1/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer1/while/Identity_3?
price_layer1/while/Identity_4Identity)price_layer1/while/lstm_cell_14/mul_2:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer1/while/Identity_4?
price_layer1/while/Identity_5Identity)price_layer1/while/lstm_cell_14/add_1:z:07^price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6^price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp8^price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer1/while/Identity_5"C
price_layer1_while_identity$price_layer1/while/Identity:output:0"G
price_layer1_while_identity_1&price_layer1/while/Identity_1:output:0"G
price_layer1_while_identity_2&price_layer1/while/Identity_2:output:0"G
price_layer1_while_identity_3&price_layer1/while/Identity_3:output:0"G
price_layer1_while_identity_4&price_layer1/while/Identity_4:output:0"G
price_layer1_while_identity_5&price_layer1/while/Identity_5:output:0"?
?price_layer1_while_lstm_cell_14_biasadd_readvariableop_resourceAprice_layer1_while_lstm_cell_14_biasadd_readvariableop_resource_0"?
@price_layer1_while_lstm_cell_14_matmul_1_readvariableop_resourceBprice_layer1_while_lstm_cell_14_matmul_1_readvariableop_resource_0"?
>price_layer1_while_lstm_cell_14_matmul_readvariableop_resource@price_layer1_while_lstm_cell_14_matmul_readvariableop_resource_0"d
/price_layer1_while_price_layer1_strided_slice_11price_layer1_while_price_layer1_strided_slice_1_0"?
kprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensormprice_layer1_while_tensorarrayv2read_tensorlistgetitem_price_layer1_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2p
6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp6price_layer1/while/lstm_cell_14/BiasAdd/ReadVariableOp2n
5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp5price_layer1/while/lstm_cell_14/MatMul/ReadVariableOp2r
7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp7price_layer1/while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?%
?
while_body_1196144953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0#
while_lstm_cell_14_1196144977_0#
while_lstm_cell_14_1196144979_0#
while_lstm_cell_14_1196144981_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor!
while_lstm_cell_14_1196144977!
while_lstm_cell_14_1196144979!
while_lstm_cell_14_1196144981??*while/lstm_cell_14/StatefulPartitionedCall?
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
*while/lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_14_1196144977_0while_lstm_cell_14_1196144979_0while_lstm_cell_14_1196144981_0*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_11961445272,
*while/lstm_cell_14/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_14/StatefulPartitionedCall:output:0*
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
while/IdentityIdentitywhile/add_1:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0+^while/lstm_cell_14/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity3while/lstm_cell_14/StatefulPartitionedCall:output:1+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identity3while/lstm_cell_14/StatefulPartitionedCall:output:2+^while/lstm_cell_14/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"@
while_lstm_cell_14_1196144977while_lstm_cell_14_1196144977_0"@
while_lstm_cell_14_1196144979while_lstm_cell_14_1196144979_0"@
while_lstm_cell_14_1196144981while_lstm_cell_14_1196144981_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2X
*while/lstm_cell_14/StatefulPartitionedCall*while/lstm_cell_14/StatefulPartitionedCall: 
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
1__inference_lstm_cell_14_layer_call_fn_1196148845

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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_11961445272
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
E:?????????:????????? :????????? :::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
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
?
"price_layer1_while_cond_11961470316
2price_layer1_while_price_layer1_while_loop_counter<
8price_layer1_while_price_layer1_while_maximum_iterations"
price_layer1_while_placeholder$
 price_layer1_while_placeholder_1$
 price_layer1_while_placeholder_2$
 price_layer1_while_placeholder_38
4price_layer1_while_less_price_layer1_strided_slice_1R
Nprice_layer1_while_price_layer1_while_cond_1196147031___redundant_placeholder0R
Nprice_layer1_while_price_layer1_while_cond_1196147031___redundant_placeholder1R
Nprice_layer1_while_price_layer1_while_cond_1196147031___redundant_placeholder2R
Nprice_layer1_while_price_layer1_while_cond_1196147031___redundant_placeholder3
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
??
?

G__inference_model_9_layer_call_and_return_conditional_losses_1196146963
inputs_0
inputs_1<
8price_layer1_lstm_cell_14_matmul_readvariableop_resource>
:price_layer1_lstm_cell_14_matmul_1_readvariableop_resource=
9price_layer1_lstm_cell_14_biasadd_readvariableop_resource<
8price_layer2_lstm_cell_15_matmul_readvariableop_resource>
:price_layer2_lstm_cell_15_matmul_1_readvariableop_resource=
9price_layer2_lstm_cell_15_biasadd_readvariableop_resource/
+fixed_layer1_matmul_readvariableop_resource0
,fixed_layer1_biasadd_readvariableop_resource/
+fixed_layer2_matmul_readvariableop_resource0
,fixed_layer2_biasadd_readvariableop_resource0
,action_output_matmul_readvariableop_resource1
-action_output_biasadd_readvariableop_resource
identity??$action_output/BiasAdd/ReadVariableOp?#action_output/MatMul/ReadVariableOp?#fixed_layer1/BiasAdd/ReadVariableOp?"fixed_layer1/MatMul/ReadVariableOp?#fixed_layer2/BiasAdd/ReadVariableOp?"fixed_layer2/MatMul/ReadVariableOp?0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp?/price_layer1/lstm_cell_14/MatMul/ReadVariableOp?1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp?price_layer1/while?0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp?/price_layer2/lstm_cell_15/MatMul/ReadVariableOp?1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp?price_layer2/while`
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
value	B : 2
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
value	B : 2
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
:????????? 2
price_layer1/zerosz
price_layer1/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B : 2
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
value	B : 2
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
:????????? 2
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
/price_layer1/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp8price_layer1_lstm_cell_14_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/price_layer1/lstm_cell_14/MatMul/ReadVariableOp?
 price_layer1/lstm_cell_14/MatMulMatMul%price_layer1/strided_slice_2:output:07price_layer1/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 price_layer1/lstm_cell_14/MatMul?
1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp:price_layer1_lstm_cell_14_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp?
"price_layer1/lstm_cell_14/MatMul_1MatMulprice_layer1/zeros:output:09price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"price_layer1/lstm_cell_14/MatMul_1?
price_layer1/lstm_cell_14/addAddV2*price_layer1/lstm_cell_14/MatMul:product:0,price_layer1/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
price_layer1/lstm_cell_14/add?
0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp9price_layer1_lstm_cell_14_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp?
!price_layer1/lstm_cell_14/BiasAddBiasAdd!price_layer1/lstm_cell_14/add:z:08price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!price_layer1/lstm_cell_14/BiasAdd?
price_layer1/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
price_layer1/lstm_cell_14/Const?
)price_layer1/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)price_layer1/lstm_cell_14/split/split_dim?
price_layer1/lstm_cell_14/splitSplit2price_layer1/lstm_cell_14/split/split_dim:output:0*price_layer1/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2!
price_layer1/lstm_cell_14/split?
!price_layer1/lstm_cell_14/SigmoidSigmoid(price_layer1/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2#
!price_layer1/lstm_cell_14/Sigmoid?
#price_layer1/lstm_cell_14/Sigmoid_1Sigmoid(price_layer1/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2%
#price_layer1/lstm_cell_14/Sigmoid_1?
price_layer1/lstm_cell_14/mulMul'price_layer1/lstm_cell_14/Sigmoid_1:y:0price_layer1/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
price_layer1/lstm_cell_14/mul?
price_layer1/lstm_cell_14/ReluRelu(price_layer1/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2 
price_layer1/lstm_cell_14/Relu?
price_layer1/lstm_cell_14/mul_1Mul%price_layer1/lstm_cell_14/Sigmoid:y:0,price_layer1/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_14/mul_1?
price_layer1/lstm_cell_14/add_1AddV2!price_layer1/lstm_cell_14/mul:z:0#price_layer1/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_14/add_1?
#price_layer1/lstm_cell_14/Sigmoid_2Sigmoid(price_layer1/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2%
#price_layer1/lstm_cell_14/Sigmoid_2?
 price_layer1/lstm_cell_14/Relu_1Relu#price_layer1/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 price_layer1/lstm_cell_14/Relu_1?
price_layer1/lstm_cell_14/mul_2Mul'price_layer1/lstm_cell_14/Sigmoid_2:y:0.price_layer1/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer1/lstm_cell_14/mul_2?
*price_layer1/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2,
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
price_layer1/whileWhile(price_layer1/while/loop_counter:output:0.price_layer1/while/maximum_iterations:output:0price_layer1/time:output:0%price_layer1/TensorArrayV2_1:handle:0price_layer1/zeros:output:0price_layer1/zeros_1:output:0%price_layer1/strided_slice_1:output:0Dprice_layer1/TensorArrayUnstack/TensorListFromTensor:output_handle:08price_layer1_lstm_cell_14_matmul_readvariableop_resource:price_layer1_lstm_cell_14_matmul_1_readvariableop_resource9price_layer1_lstm_cell_14_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"price_layer1_while_body_1196146705*.
cond&R$
"price_layer1_while_cond_1196146704*K
output_shapes:
8: : : : :????????? :????????? : : : : : *
parallel_iterations 2
price_layer1/while?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2?
=price_layer1/TensorArrayV2Stack/TensorListStack/element_shape?
/price_layer1/TensorArrayV2Stack/TensorListStackTensorListStackprice_layer1/while:output:3Fprice_layer1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:????????? *
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
:????????? *
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
:????????? 2
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
:????????? 2
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
valueB"????    2D
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
:????????? *
shrink_axis_mask2
price_layer2/strided_slice_2?
/price_layer2/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp8price_layer2_lstm_cell_15_matmul_readvariableop_resource*
_output_shapes
:	 ?*
dtype021
/price_layer2/lstm_cell_15/MatMul/ReadVariableOp?
 price_layer2/lstm_cell_15/MatMulMatMul%price_layer2/strided_slice_2:output:07price_layer2/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2"
 price_layer2/lstm_cell_15/MatMul?
1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOp:price_layer2_lstm_cell_15_matmul_1_readvariableop_resource*
_output_shapes
:	 ?*
dtype023
1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp?
"price_layer2/lstm_cell_15/MatMul_1MatMulprice_layer2/zeros:output:09price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"price_layer2/lstm_cell_15/MatMul_1?
price_layer2/lstm_cell_15/addAddV2*price_layer2/lstm_cell_15/MatMul:product:0,price_layer2/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
price_layer2/lstm_cell_15/add?
0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOp9price_layer2_lstm_cell_15_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype022
0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp?
!price_layer2/lstm_cell_15/BiasAddBiasAdd!price_layer2/lstm_cell_15/add:z:08price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2#
!price_layer2/lstm_cell_15/BiasAdd?
price_layer2/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
price_layer2/lstm_cell_15/Const?
)price_layer2/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)price_layer2/lstm_cell_15/split/split_dim?
price_layer2/lstm_cell_15/splitSplit2price_layer2/lstm_cell_15/split/split_dim:output:0*price_layer2/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2!
price_layer2/lstm_cell_15/split?
!price_layer2/lstm_cell_15/SigmoidSigmoid(price_layer2/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2#
!price_layer2/lstm_cell_15/Sigmoid?
#price_layer2/lstm_cell_15/Sigmoid_1Sigmoid(price_layer2/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2%
#price_layer2/lstm_cell_15/Sigmoid_1?
price_layer2/lstm_cell_15/mulMul'price_layer2/lstm_cell_15/Sigmoid_1:y:0price_layer2/zeros_1:output:0*
T0*'
_output_shapes
:????????? 2
price_layer2/lstm_cell_15/mul?
price_layer2/lstm_cell_15/ReluRelu(price_layer2/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2 
price_layer2/lstm_cell_15/Relu?
price_layer2/lstm_cell_15/mul_1Mul%price_layer2/lstm_cell_15/Sigmoid:y:0,price_layer2/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_15/mul_1?
price_layer2/lstm_cell_15/add_1AddV2!price_layer2/lstm_cell_15/mul:z:0#price_layer2/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_15/add_1?
#price_layer2/lstm_cell_15/Sigmoid_2Sigmoid(price_layer2/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2%
#price_layer2/lstm_cell_15/Sigmoid_2?
 price_layer2/lstm_cell_15/Relu_1Relu#price_layer2/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2"
 price_layer2/lstm_cell_15/Relu_1?
price_layer2/lstm_cell_15/mul_2Mul'price_layer2/lstm_cell_15/Sigmoid_2:y:0.price_layer2/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2!
price_layer2/lstm_cell_15/mul_2?
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
price_layer2/whileWhile(price_layer2/while/loop_counter:output:0.price_layer2/while/maximum_iterations:output:0price_layer2/time:output:0%price_layer2/TensorArrayV2_1:handle:0price_layer2/zeros:output:0price_layer2/zeros_1:output:0%price_layer2/strided_slice_1:output:0Dprice_layer2/TensorArrayUnstack/TensorListFromTensor:output_handle:08price_layer2_lstm_cell_15_matmul_readvariableop_resource:price_layer2_lstm_cell_15_matmul_1_readvariableop_resource9price_layer2_lstm_cell_15_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*.
body&R$
"price_layer2_while_body_1196146854*.
cond&R$
"price_layer2_while_cond_1196146853*K
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
IdentityIdentityaction_output/BiasAdd:output:0%^action_output/BiasAdd/ReadVariableOp$^action_output/MatMul/ReadVariableOp$^fixed_layer1/BiasAdd/ReadVariableOp#^fixed_layer1/MatMul/ReadVariableOp$^fixed_layer2/BiasAdd/ReadVariableOp#^fixed_layer2/MatMul/ReadVariableOp1^price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp0^price_layer1/lstm_cell_14/MatMul/ReadVariableOp2^price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp^price_layer1/while1^price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp0^price_layer2/lstm_cell_15/MatMul/ReadVariableOp2^price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp^price_layer2/while*
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
"fixed_layer2/MatMul/ReadVariableOp"fixed_layer2/MatMul/ReadVariableOp2d
0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp0price_layer1/lstm_cell_14/BiasAdd/ReadVariableOp2b
/price_layer1/lstm_cell_14/MatMul/ReadVariableOp/price_layer1/lstm_cell_14/MatMul/ReadVariableOp2f
1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp1price_layer1/lstm_cell_14/MatMul_1/ReadVariableOp2(
price_layer1/whileprice_layer1/while2d
0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp0price_layer2/lstm_cell_15/BiasAdd/ReadVariableOp2b
/price_layer2/lstm_cell_15/MatMul/ReadVariableOp/price_layer2/lstm_cell_15/MatMul/ReadVariableOp2f
1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp1price_layer2/lstm_cell_15/MatMul_1/ReadVariableOp2(
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
while_cond_1196148401
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196148401___redundant_placeholder08
4while_while_cond_1196148401___redundant_placeholder18
4while_while_cond_1196148401___redundant_placeholder28
4while_while_cond_1196148401___redundant_placeholder3
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
?E
?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196145022

inputs
lstm_cell_14_1196144940
lstm_cell_14_1196144942
lstm_cell_14_1196144944
identity??$lstm_cell_14/StatefulPartitionedCall?whileD
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
$lstm_cell_14/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_14_1196144940lstm_cell_14_1196144942lstm_cell_14_1196144944*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_11961445272&
$lstm_cell_14/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_14_1196144940lstm_cell_14_1196144942lstm_cell_14_1196144944*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196144953*!
condR
while_cond_1196144952*K
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
IdentityIdentitytranspose_1:y:0%^lstm_cell_14/StatefulPartitionedCall^while*
T0*4
_output_shapes"
 :?????????????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:??????????????????:::2L
$lstm_cell_14/StatefulPartitionedCall$lstm_cell_14/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :??????????????????
 
_user_specified_nameinputs
?
?
1__inference_price_layer1_layer_call_fn_1196147995

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
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961457982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?E
?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196145632

inputs
lstm_cell_15_1196145550
lstm_cell_15_1196145552
lstm_cell_15_1196145554
identity??$lstm_cell_15/StatefulPartitionedCall?whileD
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
 :?????????????????? 2
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
valueB"????    27
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
:????????? *
shrink_axis_mask2
strided_slice_2?
$lstm_cell_15/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_15_1196145550lstm_cell_15_1196145552lstm_cell_15_1196145554*
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
GPU 2J 8? *U
fPRN
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_11961451372&
$lstm_cell_15/StatefulPartitionedCall?
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_15_1196145550lstm_cell_15_1196145552lstm_cell_15_1196145554*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :????????? :????????? : : : : : *%
_read_only_resource_inputs
	
*!
bodyR
while_body_1196145563*!
condR
while_cond_1196145562*K
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
IdentityIdentitystrided_slice_3:output:0%^lstm_cell_15/StatefulPartitionedCall^while*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????????????? :::2L
$lstm_cell_15/StatefulPartitionedCall$lstm_cell_15/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
?V
?
"price_layer2_while_body_11961468546
2price_layer2_while_price_layer2_while_loop_counter<
8price_layer2_while_price_layer2_while_maximum_iterations"
price_layer2_while_placeholder$
 price_layer2_while_placeholder_1$
 price_layer2_while_placeholder_2$
 price_layer2_while_placeholder_35
1price_layer2_while_price_layer2_strided_slice_1_0q
mprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0D
@price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0F
Bprice_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0E
Aprice_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0
price_layer2_while_identity!
price_layer2_while_identity_1!
price_layer2_while_identity_2!
price_layer2_while_identity_3!
price_layer2_while_identity_4!
price_layer2_while_identity_53
/price_layer2_while_price_layer2_strided_slice_1o
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensorB
>price_layer2_while_lstm_cell_15_matmul_readvariableop_resourceD
@price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resourceC
?price_layer2_while_lstm_cell_15_biasadd_readvariableop_resource??6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp?5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp?7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2F
Dprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape?
6price_layer2/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemmprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0price_layer2_while_placeholderMprice_layer2/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:????????? *
element_dtype028
6price_layer2/while/TensorArrayV2Read/TensorListGetItem?
5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOpReadVariableOp@price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype027
5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp?
&price_layer2/while/lstm_cell_15/MatMulMatMul=price_layer2/while/TensorArrayV2Read/TensorListGetItem:item:0=price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2(
&price_layer2/while/lstm_cell_15/MatMul?
7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOpReadVariableOpBprice_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype029
7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp?
(price_layer2/while/lstm_cell_15/MatMul_1MatMul price_layer2_while_placeholder_2?price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(price_layer2/while/lstm_cell_15/MatMul_1?
#price_layer2/while/lstm_cell_15/addAddV20price_layer2/while/lstm_cell_15/MatMul:product:02price_layer2/while/lstm_cell_15/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2%
#price_layer2/while/lstm_cell_15/add?
6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOpReadVariableOpAprice_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype028
6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp?
'price_layer2/while/lstm_cell_15/BiasAddBiasAdd'price_layer2/while/lstm_cell_15/add:z:0>price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'price_layer2/while/lstm_cell_15/BiasAdd?
%price_layer2/while/lstm_cell_15/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%price_layer2/while/lstm_cell_15/Const?
/price_layer2/while/lstm_cell_15/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/price_layer2/while/lstm_cell_15/split/split_dim?
%price_layer2/while/lstm_cell_15/splitSplit8price_layer2/while/lstm_cell_15/split/split_dim:output:00price_layer2/while/lstm_cell_15/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2'
%price_layer2/while/lstm_cell_15/split?
'price_layer2/while/lstm_cell_15/SigmoidSigmoid.price_layer2/while/lstm_cell_15/split:output:0*
T0*'
_output_shapes
:????????? 2)
'price_layer2/while/lstm_cell_15/Sigmoid?
)price_layer2/while/lstm_cell_15/Sigmoid_1Sigmoid.price_layer2/while/lstm_cell_15/split:output:1*
T0*'
_output_shapes
:????????? 2+
)price_layer2/while/lstm_cell_15/Sigmoid_1?
#price_layer2/while/lstm_cell_15/mulMul-price_layer2/while/lstm_cell_15/Sigmoid_1:y:0 price_layer2_while_placeholder_3*
T0*'
_output_shapes
:????????? 2%
#price_layer2/while/lstm_cell_15/mul?
$price_layer2/while/lstm_cell_15/ReluRelu.price_layer2/while/lstm_cell_15/split:output:2*
T0*'
_output_shapes
:????????? 2&
$price_layer2/while/lstm_cell_15/Relu?
%price_layer2/while/lstm_cell_15/mul_1Mul+price_layer2/while/lstm_cell_15/Sigmoid:y:02price_layer2/while/lstm_cell_15/Relu:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_15/mul_1?
%price_layer2/while/lstm_cell_15/add_1AddV2'price_layer2/while/lstm_cell_15/mul:z:0)price_layer2/while/lstm_cell_15/mul_1:z:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_15/add_1?
)price_layer2/while/lstm_cell_15/Sigmoid_2Sigmoid.price_layer2/while/lstm_cell_15/split:output:3*
T0*'
_output_shapes
:????????? 2+
)price_layer2/while/lstm_cell_15/Sigmoid_2?
&price_layer2/while/lstm_cell_15/Relu_1Relu)price_layer2/while/lstm_cell_15/add_1:z:0*
T0*'
_output_shapes
:????????? 2(
&price_layer2/while/lstm_cell_15/Relu_1?
%price_layer2/while/lstm_cell_15/mul_2Mul-price_layer2/while/lstm_cell_15/Sigmoid_2:y:04price_layer2/while/lstm_cell_15/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2'
%price_layer2/while/lstm_cell_15/mul_2?
7price_layer2/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem price_layer2_while_placeholder_1price_layer2_while_placeholder)price_layer2/while/lstm_cell_15/mul_2:z:0*
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
price_layer2/while/IdentityIdentityprice_layer2/while/add_1:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity?
price_layer2/while/Identity_1Identity8price_layer2_while_price_layer2_while_maximum_iterations7^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_1?
price_layer2/while/Identity_2Identityprice_layer2/while/add:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_2?
price_layer2/while/Identity_3IdentityGprice_layer2/while/TensorArrayV2Write/TensorListSetItem:output_handle:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
price_layer2/while/Identity_3?
price_layer2/while/Identity_4Identity)price_layer2/while/lstm_cell_15/mul_2:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
price_layer2/while/Identity_4?
price_layer2/while/Identity_5Identity)price_layer2/while/lstm_cell_15/add_1:z:07^price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6^price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp8^price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp*
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
?price_layer2_while_lstm_cell_15_biasadd_readvariableop_resourceAprice_layer2_while_lstm_cell_15_biasadd_readvariableop_resource_0"?
@price_layer2_while_lstm_cell_15_matmul_1_readvariableop_resourceBprice_layer2_while_lstm_cell_15_matmul_1_readvariableop_resource_0"?
>price_layer2_while_lstm_cell_15_matmul_readvariableop_resource@price_layer2_while_lstm_cell_15_matmul_readvariableop_resource_0"d
/price_layer2_while_price_layer2_strided_slice_11price_layer2_while_price_layer2_strided_slice_1_0"?
kprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensormprice_layer2_while_tensorarrayv2read_tensorlistgetitem_price_layer2_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2p
6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp6price_layer2/while/lstm_cell_15/BiasAdd/ReadVariableOp2n
5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp5price_layer2/while/lstm_cell_15/MatMul/ReadVariableOp2r
7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp7price_layer2/while/lstm_cell_15/MatMul_1/ReadVariableOp: 
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
?C
?
while_body_1196147418
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_07
3while_lstm_cell_14_matmul_readvariableop_resource_09
5while_lstm_cell_14_matmul_1_readvariableop_resource_08
4while_lstm_cell_14_biasadd_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor5
1while_lstm_cell_14_matmul_readvariableop_resource7
3while_lstm_cell_14_matmul_1_readvariableop_resource6
2while_lstm_cell_14_biasadd_readvariableop_resource??)while/lstm_cell_14/BiasAdd/ReadVariableOp?(while/lstm_cell_14/MatMul/ReadVariableOp?*while/lstm_cell_14/MatMul_1/ReadVariableOp?
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
(while/lstm_cell_14/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_14_matmul_readvariableop_resource_0*
_output_shapes
:	?*
dtype02*
(while/lstm_cell_14/MatMul/ReadVariableOp?
while/lstm_cell_14/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_14/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul?
*while/lstm_cell_14/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_14_matmul_1_readvariableop_resource_0*
_output_shapes
:	 ?*
dtype02,
*while/lstm_cell_14/MatMul_1/ReadVariableOp?
while/lstm_cell_14/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_14/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/MatMul_1?
while/lstm_cell_14/addAddV2#while/lstm_cell_14/MatMul:product:0%while/lstm_cell_14/MatMul_1:product:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/add?
)while/lstm_cell_14/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_14_biasadd_readvariableop_resource_0*
_output_shapes	
:?*
dtype02+
)while/lstm_cell_14/BiasAdd/ReadVariableOp?
while/lstm_cell_14/BiasAddBiasAddwhile/lstm_cell_14/add:z:01while/lstm_cell_14/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
while/lstm_cell_14/BiasAddv
while/lstm_cell_14/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell_14/Const?
"while/lstm_cell_14/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_14/split/split_dim?
while/lstm_cell_14/splitSplit+while/lstm_cell_14/split/split_dim:output:0#while/lstm_cell_14/BiasAdd:output:0*
T0*`
_output_shapesN
L:????????? :????????? :????????? :????????? *
	num_split2
while/lstm_cell_14/split?
while/lstm_cell_14/SigmoidSigmoid!while/lstm_cell_14/split:output:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid?
while/lstm_cell_14/Sigmoid_1Sigmoid!while/lstm_cell_14/split:output:1*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_1?
while/lstm_cell_14/mulMul while/lstm_cell_14/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul?
while/lstm_cell_14/ReluRelu!while/lstm_cell_14/split:output:2*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu?
while/lstm_cell_14/mul_1Mulwhile/lstm_cell_14/Sigmoid:y:0%while/lstm_cell_14/Relu:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_1?
while/lstm_cell_14/add_1AddV2while/lstm_cell_14/mul:z:0while/lstm_cell_14/mul_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/add_1?
while/lstm_cell_14/Sigmoid_2Sigmoid!while/lstm_cell_14/split:output:3*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Sigmoid_2?
while/lstm_cell_14/Relu_1Reluwhile/lstm_cell_14/add_1:z:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/Relu_1?
while/lstm_cell_14/mul_2Mul while/lstm_cell_14/Sigmoid_2:y:0'while/lstm_cell_14/Relu_1:activations:0*
T0*'
_output_shapes
:????????? 2
while/lstm_cell_14/mul_2?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_14/mul_2:z:0*
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
while/IdentityIdentitywhile/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/lstm_cell_14/mul_2:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_4?
while/Identity_5Identitywhile/lstm_cell_14/add_1:z:0*^while/lstm_cell_14/BiasAdd/ReadVariableOp)^while/lstm_cell_14/MatMul/ReadVariableOp+^while/lstm_cell_14/MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:????????? 2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"j
2while_lstm_cell_14_biasadd_readvariableop_resource4while_lstm_cell_14_biasadd_readvariableop_resource_0"l
3while_lstm_cell_14_matmul_1_readvariableop_resource5while_lstm_cell_14_matmul_1_readvariableop_resource_0"h
1while_lstm_cell_14_matmul_readvariableop_resource3while_lstm_cell_14_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*Q
_input_shapes@
>: : : : :????????? :????????? : : :::2V
)while/lstm_cell_14/BiasAdd/ReadVariableOp)while/lstm_cell_14/BiasAdd/ReadVariableOp2T
(while/lstm_cell_14/MatMul/ReadVariableOp(while/lstm_cell_14/MatMul/ReadVariableOp2X
*while/lstm_cell_14/MatMul_1/ReadVariableOp*while/lstm_cell_14/MatMul_1/ReadVariableOp: 
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
?
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196145104

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
:	 ?*
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
E:????????? :????????? :????????? :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates:OK
'
_output_shapes
:????????? 
 
_user_specified_namestates
?	
?
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1196146357

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
?&
?
G__inference_model_9_layer_call_and_return_conditional_losses_1196146503

inputs
inputs_1
price_layer1_1196146471
price_layer1_1196146473
price_layer1_1196146475
price_layer2_1196146478
price_layer2_1196146480
price_layer2_1196146482
fixed_layer1_1196146487
fixed_layer1_1196146489
fixed_layer2_1196146492
fixed_layer2_1196146494
action_output_1196146497
action_output_1196146499
identity??%action_output/StatefulPartitionedCall?$fixed_layer1/StatefulPartitionedCall?$fixed_layer2/StatefulPartitionedCall?$price_layer1/StatefulPartitionedCall?$price_layer2/StatefulPartitionedCall?
$price_layer1/StatefulPartitionedCallStatefulPartitionedCallinputsprice_layer1_1196146471price_layer1_1196146473price_layer1_1196146475*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_price_layer1_layer_call_and_return_conditional_losses_11961457982&
$price_layer1/StatefulPartitionedCall?
$price_layer2/StatefulPartitionedCallStatefulPartitionedCall-price_layer1/StatefulPartitionedCall:output:0price_layer2_1196146478price_layer2_1196146480price_layer2_1196146482*
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
GPU 2J 8? *U
fPRN
L__inference_price_layer2_layer_call_and_return_conditional_losses_11961461332&
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_11961463222
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_11961463372
concat_layer/PartitionedCall?
$fixed_layer1/StatefulPartitionedCallStatefulPartitionedCall%concat_layer/PartitionedCall:output:0fixed_layer1_1196146487fixed_layer1_1196146489*
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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_11961463572&
$fixed_layer1/StatefulPartitionedCall?
$fixed_layer2/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer1/StatefulPartitionedCall:output:0fixed_layer2_1196146492fixed_layer2_1196146494*
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_11961463842&
$fixed_layer2/StatefulPartitionedCall?
%action_output/StatefulPartitionedCallStatefulPartitionedCall-fixed_layer2/StatefulPartitionedCall:output:0action_output_1196146497action_output_1196146499*
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
M__inference_action_output_layer_call_and_return_conditional_losses_11961464102'
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
while_cond_1196146200
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_18
4while_while_cond_1196146200___redundant_placeholder08
4while_while_cond_1196146200___redundant_placeholder18
4while_while_cond_1196146200___redundant_placeholder28
4while_while_cond_1196146200___redundant_placeholder3
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
:"?L
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
?__call__"?M
_tf_keras_network?M{"class_name": "Functional", "name": "model_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5, 1]}, {"class_name": "TensorShape", "items": [null, 2]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_9", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "price_input"}, "name": "price_input", "inbound_nodes": []}, {"class_name": "LSTM", "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer1", "inbound_nodes": [[["price_input", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "name": "price_layer2", "inbound_nodes": [[["price_layer1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "price_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "price_flatten", "inbound_nodes": [[["price_layer2", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "env_input"}, "name": "env_input", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concat_layer", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concat_layer", "inbound_nodes": [[["price_flatten", 0, 0, {}], ["env_input", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer1", "trainable": true, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer1", "inbound_nodes": [[["concat_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fixed_layer2", "trainable": true, "dtype": "float32", "units": 4, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fixed_layer2", "inbound_nodes": [[["fixed_layer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "action_output", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "action_output", "inbound_nodes": [[["fixed_layer2", 0, 0, {}]]]}], "input_layers": [["price_input", 0, 0], ["env_input", 0, 0]], "output_layers": [["action_output", 0, 0]]}}, "training_config": {"loss": {"action_output": "mse"}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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

_tf_keras_rnn_layer?	{"class_name": "LSTM", "name": "price_layer1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer1", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 1]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 1]}}
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

_tf_keras_rnn_layer?
{"class_name": "LSTM", "name": "price_layer2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "price_layer2", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 32]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 5, 32]}}
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
?
7iter

8beta_1

9beta_2
	:decay
;learning_rate%m?&m?+m?,m?1m?2m?<m?=m?>m??m?@m?Am?%v?&v?+v?,v?1v?2v?<v?=v?>v??v?@v?Av?"
	optimizer
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
v
<0
=1
>2
?3
@4
A5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
v
<0
=1
>2
?3
@4
A5
%6
&7
+8
,9
110
211"
trackable_list_wrapper
?
Bmetrics
Clayer_regularization_losses

Dlayers
Enon_trainable_variables
regularization_losses
	variables
Flayer_metrics
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

<kernel
=recurrent_kernel
>bias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_14", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
?
Kmetrics
Llayer_regularization_losses

Mlayers
Nnon_trainable_variables
regularization_losses
trainable_variables
	variables
Olayer_metrics

Pstates
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

?kernel
@recurrent_kernel
Abias
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "LSTMCell", "name": "lstm_cell_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell_15", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
?
Umetrics
Vlayer_regularization_losses

Wlayers
Xnon_trainable_variables
regularization_losses
trainable_variables
	variables
Ylayer_metrics

Zstates
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
[layer_regularization_losses
\metrics

]layers
^non_trainable_variables
regularization_losses
	variables
_layer_metrics
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
`layer_regularization_losses
ametrics

blayers
cnon_trainable_variables
!regularization_losses
"	variables
dlayer_metrics
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
elayer_regularization_losses
fmetrics

glayers
hnon_trainable_variables
'regularization_losses
(	variables
ilayer_metrics
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
jlayer_regularization_losses
kmetrics

llayers
mnon_trainable_variables
-regularization_losses
.	variables
nlayer_metrics
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
olayer_regularization_losses
pmetrics

qlayers
rnon_trainable_variables
3regularization_losses
4	variables
slayer_metrics
5trainable_variables
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
3:1	?2 price_layer1/lstm_cell_14/kernel
=:;	 ?2*price_layer1/lstm_cell_14/recurrent_kernel
-:+?2price_layer1/lstm_cell_14/bias
3:1	 ?2 price_layer2/lstm_cell_15/kernel
=:;	 ?2*price_layer2/lstm_cell_15/recurrent_kernel
-:+?2price_layer2/lstm_cell_15/bias
'
t0"
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
<0
=1
>2"
trackable_list_wrapper
5
<0
=1
>2"
trackable_list_wrapper
?
ulayer_regularization_losses
vmetrics

wlayers
xnon_trainable_variables
Gregularization_losses
H	variables
ylayer_metrics
Itrainable_variables
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
?0
@1
A2"
trackable_list_wrapper
5
?0
@1
A2"
trackable_list_wrapper
?
zlayer_regularization_losses
{metrics

|layers
}non_trainable_variables
Qregularization_losses
R	variables
~layer_metrics
Strainable_variables
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
	total

?count
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
/
0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:("2Adam/fixed_layer1/kernel/m
$:"2Adam/fixed_layer1/bias/m
*:(2Adam/fixed_layer2/kernel/m
$:"2Adam/fixed_layer2/bias/m
+:)2Adam/action_output/kernel/m
%:#2Adam/action_output/bias/m
8:6	?2'Adam/price_layer1/lstm_cell_14/kernel/m
B:@	 ?21Adam/price_layer1/lstm_cell_14/recurrent_kernel/m
2:0?2%Adam/price_layer1/lstm_cell_14/bias/m
8:6	 ?2'Adam/price_layer2/lstm_cell_15/kernel/m
B:@	 ?21Adam/price_layer2/lstm_cell_15/recurrent_kernel/m
2:0?2%Adam/price_layer2/lstm_cell_15/bias/m
*:("2Adam/fixed_layer1/kernel/v
$:"2Adam/fixed_layer1/bias/v
*:(2Adam/fixed_layer2/kernel/v
$:"2Adam/fixed_layer2/bias/v
+:)2Adam/action_output/kernel/v
%:#2Adam/action_output/bias/v
8:6	?2'Adam/price_layer1/lstm_cell_14/kernel/v
B:@	 ?21Adam/price_layer1/lstm_cell_14/recurrent_kernel/v
2:0?2%Adam/price_layer1/lstm_cell_14/bias/v
8:6	 ?2'Adam/price_layer2/lstm_cell_15/kernel/v
B:@	 ?21Adam/price_layer2/lstm_cell_15/recurrent_kernel/v
2:0?2%Adam/price_layer2/lstm_cell_15/bias/v
?2?
G__inference_model_9_layer_call_and_return_conditional_losses_1196147290
G__inference_model_9_layer_call_and_return_conditional_losses_1196146963
G__inference_model_9_layer_call_and_return_conditional_losses_1196146463
G__inference_model_9_layer_call_and_return_conditional_losses_1196146427?
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
%__inference__wrapped_model_1196144421?
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
,__inference_model_9_layer_call_fn_1196146596
,__inference_model_9_layer_call_fn_1196147350
,__inference_model_9_layer_call_fn_1196146530
,__inference_model_9_layer_call_fn_1196147320?
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
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147503
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147831
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147656
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147984?
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
1__inference_price_layer1_layer_call_fn_1196147667
1__inference_price_layer1_layer_call_fn_1196148006
1__inference_price_layer1_layer_call_fn_1196147678
1__inference_price_layer1_layer_call_fn_1196147995?
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
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148312
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148640
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148159
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148487?
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
1__inference_price_layer2_layer_call_fn_1196148662
1__inference_price_layer2_layer_call_fn_1196148651
1__inference_price_layer2_layer_call_fn_1196148323
1__inference_price_layer2_layer_call_fn_1196148334?
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_1196148668?
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
2__inference_price_flatten_layer_call_fn_1196148673?
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
L__inference_concat_layer_layer_call_and_return_conditional_losses_1196148680?
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
1__inference_concat_layer_layer_call_fn_1196148686?
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
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1196148697?
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
1__inference_fixed_layer1_layer_call_fn_1196148706?
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
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1196148717?
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
1__inference_fixed_layer2_layer_call_fn_1196148726?
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
M__inference_action_output_layer_call_and_return_conditional_losses_1196148736?
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
2__inference_action_output_layer_call_fn_1196148745?
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
(__inference_signature_wrapper_1196146636	env_inputprice_input"?
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
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196148811
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196148778?
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
1__inference_lstm_cell_14_layer_call_fn_1196148845
1__inference_lstm_cell_14_layer_call_fn_1196148828?
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
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196148911
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196148878?
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
1__inference_lstm_cell_15_layer_call_fn_1196148928
1__inference_lstm_cell_15_layer_call_fn_1196148945?
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
%__inference__wrapped_model_1196144421?<=>?@A%&+,12b?_
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
M__inference_action_output_layer_call_and_return_conditional_losses_1196148736\12/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
2__inference_action_output_layer_call_fn_1196148745O12/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_concat_layer_layer_call_and_return_conditional_losses_1196148680?Z?W
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
1__inference_concat_layer_layer_call_fn_1196148686vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????"?
L__inference_fixed_layer1_layer_call_and_return_conditional_losses_1196148697\%&/?,
%?"
 ?
inputs?????????"
? "%?"
?
0?????????
? ?
1__inference_fixed_layer1_layer_call_fn_1196148706O%&/?,
%?"
 ?
inputs?????????"
? "???????????
L__inference_fixed_layer2_layer_call_and_return_conditional_losses_1196148717\+,/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? ?
1__inference_fixed_layer2_layer_call_fn_1196148726O+,/?,
%?"
 ?
inputs?????????
? "???????????
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196148778?<=>??}
v?s
 ?
inputs?????????
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
L__inference_lstm_cell_14_layer_call_and_return_conditional_losses_1196148811?<=>??}
v?s
 ?
inputs?????????
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
1__inference_lstm_cell_14_layer_call_fn_1196148828?<=>??}
v?s
 ?
inputs?????????
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
1__inference_lstm_cell_14_layer_call_fn_1196148845?<=>??}
v?s
 ?
inputs?????????
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
1/1????????? ?
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196148878??@A??}
v?s
 ?
inputs????????? 
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
L__inference_lstm_cell_15_layer_call_and_return_conditional_losses_1196148911??@A??}
v?s
 ?
inputs????????? 
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
1__inference_lstm_cell_15_layer_call_fn_1196148928??@A??}
v?s
 ?
inputs????????? 
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
1__inference_lstm_cell_15_layer_call_fn_1196148945??@A??}
v?s
 ?
inputs????????? 
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
G__inference_model_9_layer_call_and_return_conditional_losses_1196146427?<=>?@A%&+,12j?g
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
G__inference_model_9_layer_call_and_return_conditional_losses_1196146463?<=>?@A%&+,12j?g
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
G__inference_model_9_layer_call_and_return_conditional_losses_1196146963?<=>?@A%&+,12f?c
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
G__inference_model_9_layer_call_and_return_conditional_losses_1196147290?<=>?@A%&+,12f?c
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
,__inference_model_9_layer_call_fn_1196146530?<=>?@A%&+,12j?g
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
,__inference_model_9_layer_call_fn_1196146596?<=>?@A%&+,12j?g
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
,__inference_model_9_layer_call_fn_1196147320?<=>?@A%&+,12f?c
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
,__inference_model_9_layer_call_fn_1196147350?<=>?@A%&+,12f?c
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
M__inference_price_flatten_layer_call_and_return_conditional_losses_1196148668X/?,
%?"
 ?
inputs????????? 
? "%?"
?
0????????? 
? ?
2__inference_price_flatten_layer_call_fn_1196148673K/?,
%?"
 ?
inputs????????? 
? "?????????? ?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147503?<=>O?L
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
0?????????????????? 
? ?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147656?<=>O?L
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
0?????????????????? 
? ?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147831q<=>??<
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
0????????? 
? ?
L__inference_price_layer1_layer_call_and_return_conditional_losses_1196147984q<=>??<
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
0????????? 
? ?
1__inference_price_layer1_layer_call_fn_1196147667}<=>O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p

 
? "%?"?????????????????? ?
1__inference_price_layer1_layer_call_fn_1196147678}<=>O?L
E?B
4?1
/?,
inputs/0??????????????????

 
p 

 
? "%?"?????????????????? ?
1__inference_price_layer1_layer_call_fn_1196147995d<=>??<
5?2
$?!
inputs?????????

 
p

 
? "?????????? ?
1__inference_price_layer1_layer_call_fn_1196148006d<=>??<
5?2
$?!
inputs?????????

 
p 

 
? "?????????? ?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148159m?@A??<
5?2
$?!
inputs????????? 

 
p

 
? "%?"
?
0????????? 
? ?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148312m?@A??<
5?2
$?!
inputs????????? 

 
p 

 
? "%?"
?
0????????? 
? ?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148487}?@AO?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p

 
? "%?"
?
0????????? 
? ?
L__inference_price_layer2_layer_call_and_return_conditional_losses_1196148640}?@AO?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p 

 
? "%?"
?
0????????? 
? ?
1__inference_price_layer2_layer_call_fn_1196148323`?@A??<
5?2
$?!
inputs????????? 

 
p

 
? "?????????? ?
1__inference_price_layer2_layer_call_fn_1196148334`?@A??<
5?2
$?!
inputs????????? 

 
p 

 
? "?????????? ?
1__inference_price_layer2_layer_call_fn_1196148651p?@AO?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p

 
? "?????????? ?
1__inference_price_layer2_layer_call_fn_1196148662p?@AO?L
E?B
4?1
/?,
inputs/0?????????????????? 

 
p 

 
? "?????????? ?
(__inference_signature_wrapper_1196146636?<=>?@A%&+,12y?v
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