??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
8
Const
output"dtype"
valuetensor"
dtypetype
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
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
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
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
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
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
(recommender_net_1/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*9
shared_name*(recommender_net_1/embedding_2/embeddings
?
<recommender_net_1/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp(recommender_net_1/embedding_2/embeddings*
_output_shapes
:	?N*
dtype0
?
(recommender_net_1/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*9
shared_name*(recommender_net_1/embedding_3/embeddings
?
<recommender_net_1/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp(recommender_net_1/embedding_3/embeddings*
_output_shapes
:	?N*
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
/Adam/recommender_net_1/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*@
shared_name1/Adam/recommender_net_1/embedding_2/embeddings/m
?
CAdam/recommender_net_1/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOp/Adam/recommender_net_1/embedding_2/embeddings/m*
_output_shapes
:	?N*
dtype0
?
/Adam/recommender_net_1/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*@
shared_name1/Adam/recommender_net_1/embedding_3/embeddings/m
?
CAdam/recommender_net_1/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOp/Adam/recommender_net_1/embedding_3/embeddings/m*
_output_shapes
:	?N*
dtype0
?
/Adam/recommender_net_1/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*@
shared_name1/Adam/recommender_net_1/embedding_2/embeddings/v
?
CAdam/recommender_net_1/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOp/Adam/recommender_net_1/embedding_2/embeddings/v*
_output_shapes
:	?N*
dtype0
?
/Adam/recommender_net_1/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?N*@
shared_name1/Adam/recommender_net_1/embedding_3/embeddings/v
?
CAdam/recommender_net_1/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOp/Adam/recommender_net_1/embedding_3/embeddings/v*
_output_shapes
:	?N*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
user_embedding
item_embedding
dot
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
b


embeddings
	variables
regularization_losses
trainable_variables
	keras_api
b

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
d
iter

beta_1

beta_2
	decay
learning_rate
m6m7
v8v9


0
1
 


0
1
?
non_trainable_variables
	variables
layer_regularization_losses
metrics
regularization_losses

 layers
!layer_metrics
trainable_variables
 
rp
VARIABLE_VALUE(recommender_net_1/embedding_2/embeddings4user_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE


0
 


0
?
"non_trainable_variables
	variables
#layer_regularization_losses
$metrics
regularization_losses

%layers
&layer_metrics
trainable_variables
rp
VARIABLE_VALUE(recommender_net_1/embedding_3/embeddings4item_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
?
'non_trainable_variables
	variables
(layer_regularization_losses
)metrics
regularization_losses

*layers
+layer_metrics
trainable_variables
 
 
 
?
,non_trainable_variables
	variables
-layer_regularization_losses
.metrics
regularization_losses

/layers
0layer_metrics
trainable_variables
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
 
 

10

0
1
2
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
	2total
	3count
4	variables
5	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

20
31

4	variables
??
VARIABLE_VALUE/Adam/recommender_net_1/embedding_2/embeddings/mPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/recommender_net_1/embedding_3/embeddings/mPitem_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/recommender_net_1/embedding_2/embeddings/vPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/recommender_net_1/embedding_3/embeddings/vPitem_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1(recommender_net_1/embedding_2/embeddings(recommender_net_1/embedding_3/embeddings*
Tin
2	*
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
GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3187645
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename<recommender_net_1/embedding_2/embeddings/Read/ReadVariableOp<recommender_net_1/embedding_3/embeddings/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpCAdam/recommender_net_1/embedding_2/embeddings/m/Read/ReadVariableOpCAdam/recommender_net_1/embedding_3/embeddings/m/Read/ReadVariableOpCAdam/recommender_net_1/embedding_2/embeddings/v/Read/ReadVariableOpCAdam/recommender_net_1/embedding_3/embeddings/v/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_3187803
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename(recommender_net_1/embedding_2/embeddings(recommender_net_1/embedding_3/embeddings	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount/Adam/recommender_net_1/embedding_2/embeddings/m/Adam/recommender_net_1/embedding_3/embeddings/m/Adam/recommender_net_1/embedding_2/embeddings/v/Adam/recommender_net_1/embedding_3/embeddings/v*
Tin
2*
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
#__inference__traced_restore_3187852??
?
?
H__inference_embedding_3_layer_call_and_return_conditional_losses_3187550

inputs	+
embedding_lookup_3187538:	?N
identity??embedding_lookup?Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookupResourceGatherembedding_lookup_3187538inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/3187538*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3187538*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3187538*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_3/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_3/embeddings/Regularizer/Square?
:recommender_net_1/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_3/embeddings/Regularizer/Const?
8recommender_net_1/embedding_3/embeddings/Regularizer/SumSum?recommender_net_1/embedding_3/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_3/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_3/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookupK^recommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup2?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_embedding_2_layer_call_and_return_conditional_losses_3187525

inputs	+
embedding_lookup_3187513:	?N
identity??embedding_lookup?Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookupResourceGatherembedding_lookup_3187513inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/3187513*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3187513*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3187513*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_2/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_2/embeddings/Regularizer/Square?
:recommender_net_1/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_2/embeddings/Regularizer/Const?
8recommender_net_1/embedding_2/embeddings/Regularizer/SumSum?recommender_net_1/embedding_2/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_2/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_2/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookupK^recommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup2?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
#__inference__traced_restore_3187852
file_prefixL
9assignvariableop_recommender_net_1_embedding_2_embeddings:	?NN
;assignvariableop_1_recommender_net_1_embedding_3_embeddings:	?N&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: U
Bassignvariableop_9_adam_recommender_net_1_embedding_2_embeddings_m:	?NV
Cassignvariableop_10_adam_recommender_net_1_embedding_3_embeddings_m:	?NV
Cassignvariableop_11_adam_recommender_net_1_embedding_2_embeddings_v:	?NV
Cassignvariableop_12_adam_recommender_net_1_embedding_3_embeddings_v:	?N
identity_14??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B4user_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB4item_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPitem_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPitem_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp9assignvariableop_recommender_net_1_embedding_2_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp;assignvariableop_1_recommender_net_1_embedding_3_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpBassignvariableop_9_adam_recommender_net_1_embedding_2_embeddings_mIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpCassignvariableop_10_adam_recommender_net_1_embedding_3_embeddings_mIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpCassignvariableop_11_adam_recommender_net_1_embedding_2_embeddings_vIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpCassignvariableop_12_adam_recommender_net_1_embedding_3_embeddings_vIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_13Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_13?
Identity_14IdentityIdentity_13:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_14"#
identity_14Identity_14:output:0*/
_input_shapes
: : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122(
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
?
?
%__inference_signature_wrapper_3187645
input_1	
unknown:	?N
	unknown_0:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2	*
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
GPU 2J 8? *+
f&R$
"__inference__wrapped_model_31875022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
3__inference_recommender_net_1_layer_call_fn_3187593
input_1	
unknown:	?N
	unknown_0:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2	*
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
GPU 2J 8? *W
fRRP
N__inference_recommender_net_1_layer_call_and_return_conditional_losses_31875832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?(
?
 __inference__traced_save_3187803
file_prefixG
Csavev2_recommender_net_1_embedding_2_embeddings_read_readvariableopG
Csavev2_recommender_net_1_embedding_3_embeddings_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopN
Jsavev2_adam_recommender_net_1_embedding_2_embeddings_m_read_readvariableopN
Jsavev2_adam_recommender_net_1_embedding_3_embeddings_m_read_readvariableopN
Jsavev2_adam_recommender_net_1_embedding_2_embeddings_v_read_readvariableopN
Jsavev2_adam_recommender_net_1_embedding_3_embeddings_v_read_readvariableop
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
:*
dtype0*?
value?B?B4user_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB4item_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPitem_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPitem_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Csavev2_recommender_net_1_embedding_2_embeddings_read_readvariableopCsavev2_recommender_net_1_embedding_3_embeddings_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopJsavev2_adam_recommender_net_1_embedding_2_embeddings_m_read_readvariableopJsavev2_adam_recommender_net_1_embedding_3_embeddings_m_read_readvariableopJsavev2_adam_recommender_net_1_embedding_2_embeddings_v_read_readvariableopJsavev2_adam_recommender_net_1_embedding_3_embeddings_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
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

identity_1Identity_1:output:0*g
_input_shapesV
T: :	?N:	?N: : : : : : : :	?N:	?N:	?N:	?N: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?N:%!

_output_shapes
:	?N:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :%
!

_output_shapes
:	?N:%!

_output_shapes
:	?N:%!

_output_shapes
:	?N:%!

_output_shapes
:	?N:

_output_shapes
: 
?
?
H__inference_embedding_3_layer_call_and_return_conditional_losses_3187694

inputs	+
embedding_lookup_3187682:	?N
identity??embedding_lookup?Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookupResourceGatherembedding_lookup_3187682inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/3187682*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3187682*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3187682*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_3/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_3/embeddings/Regularizer/Square?
:recommender_net_1/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_3/embeddings/Regularizer/Const?
8recommender_net_1/embedding_3/embeddings/Regularizer/SumSum?recommender_net_1/embedding_3/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_3/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_3/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookupK^recommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup2?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_3187730f
Srecommender_net_1_embedding_2_embeddings_regularizer_square_readvariableop_resource:	?N
identity??Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpSrecommender_net_1_embedding_2_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_2/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_2/embeddings/Regularizer/Square?
:recommender_net_1/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_2/embeddings/Regularizer/Const?
8recommender_net_1/embedding_2/embeddings/Regularizer/SumSum?recommender_net_1/embedding_2/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_2/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_2/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/mul?
IdentityIdentity<recommender_net_1/embedding_2/embeddings/Regularizer/mul:z:0K^recommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp
?
?
-__inference_embedding_3_layer_call_fn_3187701

inputs	
unknown:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_3_layer_call_and_return_conditional_losses_31875502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_3187741f
Srecommender_net_1_embedding_3_embeddings_regularizer_square_readvariableop_resource:	?N
identity??Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpSrecommender_net_1_embedding_3_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_3/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_3/embeddings/Regularizer/Square?
:recommender_net_1/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_3/embeddings/Regularizer/Const?
8recommender_net_1/embedding_3/embeddings/Regularizer/SumSum?recommender_net_1/embedding_3/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_3/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_3/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/mul?
IdentityIdentity<recommender_net_1/embedding_3/embeddings/Regularizer/mul:z:0K^recommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp
?

n
B__inference_dot_1_layer_call_and_return_conditional_losses_3187713
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1?
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapew
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
H__inference_embedding_2_layer_call_and_return_conditional_losses_3187666

inputs	+
embedding_lookup_3187654:	?N
identity??embedding_lookup?Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookupResourceGatherembedding_lookup_3187654inputs",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*+
_class!
loc:@embedding_lookup/3187654*'
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3187654*'
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity_1?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3187654*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_2/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_2/embeddings/Regularizer/Square?
:recommender_net_1/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_2/embeddings/Regularizer/Const?
8recommender_net_1/embedding_2/embeddings/Regularizer/SumSum?recommender_net_1/embedding_2/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_2/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_2/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0^embedding_lookupK^recommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup2?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?5
?
"__inference__wrapped_model_3187502
input_1	I
6recommender_net_1_embedding_2_embedding_lookup_3187476:	?NI
6recommender_net_1_embedding_3_embedding_lookup_3187487:	?N
identity??.recommender_net_1/embedding_2/embedding_lookup?.recommender_net_1/embedding_3/embedding_lookup?
%recommender_net_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%recommender_net_1/strided_slice/stack?
'recommender_net_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2)
'recommender_net_1/strided_slice/stack_1?
'recommender_net_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'recommender_net_1/strided_slice/stack_2?
recommender_net_1/strided_sliceStridedSliceinput_1.recommender_net_1/strided_slice/stack:output:00recommender_net_1/strided_slice/stack_1:output:00recommender_net_1/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2!
recommender_net_1/strided_slice?
.recommender_net_1/embedding_2/embedding_lookupResourceGather6recommender_net_1_embedding_2_embedding_lookup_3187476(recommender_net_1/strided_slice:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*I
_class?
=;loc:@recommender_net_1/embedding_2/embedding_lookup/3187476*'
_output_shapes
:?????????*
dtype020
.recommender_net_1/embedding_2/embedding_lookup?
7recommender_net_1/embedding_2/embedding_lookup/IdentityIdentity7recommender_net_1/embedding_2/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@recommender_net_1/embedding_2/embedding_lookup/3187476*'
_output_shapes
:?????????29
7recommender_net_1/embedding_2/embedding_lookup/Identity?
9recommender_net_1/embedding_2/embedding_lookup/Identity_1Identity@recommender_net_1/embedding_2/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2;
9recommender_net_1/embedding_2/embedding_lookup/Identity_1?
(recommender_net_1/embedding_2/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2*
(recommender_net_1/embedding_2/NotEqual/y?
&recommender_net_1/embedding_2/NotEqualNotEqual(recommender_net_1/strided_slice:output:01recommender_net_1/embedding_2/NotEqual/y:output:0*
T0	*#
_output_shapes
:?????????2(
&recommender_net_1/embedding_2/NotEqual?
'recommender_net_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2)
'recommender_net_1/strided_slice_1/stack?
)recommender_net_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2+
)recommender_net_1/strided_slice_1/stack_1?
)recommender_net_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)recommender_net_1/strided_slice_1/stack_2?
!recommender_net_1/strided_slice_1StridedSliceinput_10recommender_net_1/strided_slice_1/stack:output:02recommender_net_1/strided_slice_1/stack_1:output:02recommender_net_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2#
!recommender_net_1/strided_slice_1?
.recommender_net_1/embedding_3/embedding_lookupResourceGather6recommender_net_1_embedding_3_embedding_lookup_3187487*recommender_net_1/strided_slice_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0	*I
_class?
=;loc:@recommender_net_1/embedding_3/embedding_lookup/3187487*'
_output_shapes
:?????????*
dtype020
.recommender_net_1/embedding_3/embedding_lookup?
7recommender_net_1/embedding_3/embedding_lookup/IdentityIdentity7recommender_net_1/embedding_3/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*I
_class?
=;loc:@recommender_net_1/embedding_3/embedding_lookup/3187487*'
_output_shapes
:?????????29
7recommender_net_1/embedding_3/embedding_lookup/Identity?
9recommender_net_1/embedding_3/embedding_lookup/Identity_1Identity@recommender_net_1/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2;
9recommender_net_1/embedding_3/embedding_lookup/Identity_1?
(recommender_net_1/embedding_3/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2*
(recommender_net_1/embedding_3/NotEqual/y?
&recommender_net_1/embedding_3/NotEqualNotEqual*recommender_net_1/strided_slice_1:output:01recommender_net_1/embedding_3/NotEqual/y:output:0*
T0	*#
_output_shapes
:?????????2(
&recommender_net_1/embedding_3/NotEqual?
&recommender_net_1/dot_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2(
&recommender_net_1/dot_1/ExpandDims/dim?
"recommender_net_1/dot_1/ExpandDims
ExpandDimsBrecommender_net_1/embedding_2/embedding_lookup/Identity_1:output:0/recommender_net_1/dot_1/ExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2$
"recommender_net_1/dot_1/ExpandDims?
(recommender_net_1/dot_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(recommender_net_1/dot_1/ExpandDims_1/dim?
$recommender_net_1/dot_1/ExpandDims_1
ExpandDimsBrecommender_net_1/embedding_3/embedding_lookup/Identity_1:output:01recommender_net_1/dot_1/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2&
$recommender_net_1/dot_1/ExpandDims_1?
recommender_net_1/dot_1/MatMulBatchMatMulV2+recommender_net_1/dot_1/ExpandDims:output:0-recommender_net_1/dot_1/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2 
recommender_net_1/dot_1/MatMul?
recommender_net_1/dot_1/ShapeShape'recommender_net_1/dot_1/MatMul:output:0*
T0*
_output_shapes
:2
recommender_net_1/dot_1/Shape?
recommender_net_1/dot_1/SqueezeSqueeze'recommender_net_1/dot_1/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2!
recommender_net_1/dot_1/Squeeze?
IdentityIdentity(recommender_net_1/dot_1/Squeeze:output:0/^recommender_net_1/embedding_2/embedding_lookup/^recommender_net_1/embedding_3/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2`
.recommender_net_1/embedding_2/embedding_lookup.recommender_net_1/embedding_2/embedding_lookup2`
.recommender_net_1/embedding_3/embedding_lookup.recommender_net_1/embedding_3/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
S
'__inference_dot_1_layer_call_fn_3187719
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
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_31875682
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?:
?
N__inference_recommender_net_1_layer_call_and_return_conditional_losses_3187583
input_1	&
embedding_2_3187526:	?N&
embedding_3_3187551:	?N
identity??#embedding_2/StatefulPartitionedCall?#embedding_3/StatefulPartitionedCall?Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_2_3187526*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_2_layer_call_and_return_conditional_losses_31875252%
#embedding_2/StatefulPartitionedCallr
embedding_2/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
embedding_2/NotEqual/y?
embedding_2/NotEqualNotEqualstrided_slice:output:0embedding_2/NotEqual/y:output:0*
T0	*#
_output_shapes
:?????????2
embedding_2/NotEqual
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_3_3187551*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_3_layer_call_and_return_conditional_losses_31875502%
#embedding_3/StatefulPartitionedCallr
embedding_3/NotEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
embedding_3/NotEqual/y?
embedding_3/NotEqualNotEqualstrided_slice_1:output:0embedding_3/NotEqual/y:output:0*
T0	*#
_output_shapes
:?????????2
embedding_3/NotEqual?
dot_1/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dot_1_layer_call_and_return_conditional_losses_31875682
dot_1/PartitionedCall?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_3187526*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_2/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_2/embeddings/Regularizer/Square?
:recommender_net_1/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_2/embeddings/Regularizer/Const?
8recommender_net_1/embedding_2/embeddings/Regularizer/SumSum?recommender_net_1/embedding_2/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_2/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_2/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_2/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_2/embeddings/Regularizer/mul?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_3187551*
_output_shapes
:	?N*
dtype02L
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp?
;recommender_net_1/embedding_3/embeddings/Regularizer/SquareSquareRrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?N2=
;recommender_net_1/embedding_3/embeddings/Regularizer/Square?
:recommender_net_1/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2<
:recommender_net_1/embedding_3/embeddings/Regularizer/Const?
8recommender_net_1/embedding_3/embeddings/Regularizer/SumSum?recommender_net_1/embedding_3/embeddings/Regularizer/Square:y:0Crecommender_net_1/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/Sum?
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52<
:recommender_net_1/embedding_3/embeddings/Regularizer/mul/x?
8recommender_net_1/embedding_3/embeddings/Regularizer/mulMulCrecommender_net_1/embedding_3/embeddings/Regularizer/mul/x:output:0Arecommender_net_1/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2:
8recommender_net_1/embedding_3/embeddings/Regularizer/mul?
IdentityIdentitydot_1/PartitionedCall:output:0$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallK^recommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpK^recommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2?
Jrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2?
Jrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOpJrecommender_net_1/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

l
B__inference_dot_1_layer_call_and_return_conditional_losses_3187568

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim}

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:?????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:?????????2
ExpandDims_1?
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapew
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_embedding_2_layer_call_fn_3187673

inputs	
unknown:	?N
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_embedding_2_layer_call_and_return_conditional_losses_31875252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*$
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0	?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?\
?

user_embedding
item_embedding
dot
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
*:&call_and_return_all_conditional_losses
;__call__
<_default_save_signature"?
_tf_keras_model?{"name": "recommender_net_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "RecommenderNet", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 2]}, "int64", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "RecommenderNet"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 0}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 4.999999873689376e-06, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


embeddings
	variables
regularization_losses
trainable_variables
	keras_api
*=&call_and_return_all_conditional_losses
>__call__"?
_tf_keras_layer?{"name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10000, "output_dim": 25, "embeddings_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 2}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
?

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
*?&call_and_return_all_conditional_losses
@__call__"?
_tf_keras_layer?{"name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 10000, "output_dim": 25, "embeddings_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 4}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 5}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "shared_object_id": 6, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"?
_tf_keras_layer?{"name": "dot_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dot", "config": {"name": "dot_1", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "shared_object_id": 7, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 25]}, {"class_name": "TensorShape", "items": [null, 25]}]}
w
iter

beta_1

beta_2
	decay
learning_rate
m6m7
v8v9"
	optimizer
.

0
1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
non_trainable_variables
	variables
layer_regularization_losses
metrics
regularization_losses

 layers
!layer_metrics
trainable_variables
;__call__
<_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
Eserving_default"
signature_map
;:9	?N2(recommender_net_1/embedding_2/embeddings
'

0"
trackable_list_wrapper
'
C0"
trackable_list_wrapper
'

0"
trackable_list_wrapper
?
"non_trainable_variables
	variables
#layer_regularization_losses
$metrics
regularization_losses

%layers
&layer_metrics
trainable_variables
>__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
;:9	?N2(recommender_net_1/embedding_3/embeddings
'
0"
trackable_list_wrapper
'
D0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
'non_trainable_variables
	variables
(layer_regularization_losses
)metrics
regularization_losses

*layers
+layer_metrics
trainable_variables
@__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables
	variables
-layer_regularization_losses
.metrics
regularization_losses

/layers
0layer_metrics
trainable_variables
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
D0"
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
	2total
	3count
4	variables
5	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 8}
:  (2total
:  (2count
.
20
31"
trackable_list_wrapper
-
4	variables"
_generic_user_object
@:>	?N2/Adam/recommender_net_1/embedding_2/embeddings/m
@:>	?N2/Adam/recommender_net_1/embedding_3/embeddings/m
@:>	?N2/Adam/recommender_net_1/embedding_2/embeddings/v
@:>	?N2/Adam/recommender_net_1/embedding_3/embeddings/v
?2?
N__inference_recommender_net_1_layer_call_and_return_conditional_losses_3187583?
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
annotations? *&?#
!?
input_1?????????	
?2?
3__inference_recommender_net_1_layer_call_fn_3187593?
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
annotations? *&?#
!?
input_1?????????	
?2?
"__inference__wrapped_model_3187502?
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
annotations? *&?#
!?
input_1?????????	
?2?
H__inference_embedding_2_layer_call_and_return_conditional_losses_3187666?
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
-__inference_embedding_2_layer_call_fn_3187673?
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
H__inference_embedding_3_layer_call_and_return_conditional_losses_3187694?
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
-__inference_embedding_3_layer_call_fn_3187701?
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
B__inference_dot_1_layer_call_and_return_conditional_losses_3187713?
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
'__inference_dot_1_layer_call_fn_3187719?
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
__inference_loss_fn_0_3187730?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_3187741?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
%__inference_signature_wrapper_3187645input_1"?
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
"__inference__wrapped_model_3187502k
0?-
&?#
!?
input_1?????????	
? "3?0
.
output_1"?
output_1??????????
B__inference_dot_1_layer_call_and_return_conditional_losses_3187713?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
'__inference_dot_1_layer_call_fn_3187719vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
H__inference_embedding_2_layer_call_and_return_conditional_losses_3187666W
+?(
!?
?
inputs?????????	
? "%?"
?
0?????????
? {
-__inference_embedding_2_layer_call_fn_3187673J
+?(
!?
?
inputs?????????	
? "???????????
H__inference_embedding_3_layer_call_and_return_conditional_losses_3187694W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????
? {
-__inference_embedding_3_layer_call_fn_3187701J+?(
!?
?
inputs?????????	
? "??????????<
__inference_loss_fn_0_3187730
?

? 
? "? <
__inference_loss_fn_1_3187741?

? 
? "? ?
N__inference_recommender_net_1_layer_call_and_return_conditional_losses_3187583]
0?-
&?#
!?
input_1?????????	
? "%?"
?
0?????????
? ?
3__inference_recommender_net_1_layer_call_fn_3187593P
0?-
&?#
!?
input_1?????????	
? "???????????
%__inference_signature_wrapper_3187645v
;?8
? 
1?.
,
input_1!?
input_1?????????	"3?0
.
output_1"?
output_1?????????