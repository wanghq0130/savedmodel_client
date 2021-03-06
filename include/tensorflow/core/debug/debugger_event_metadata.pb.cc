// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/debug/debugger_event_metadata.proto

#include "tensorflow/core/debug/debugger_event_metadata.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)

namespace third_party {
namespace tensorflow {
namespace core {
namespace debug {
class DebuggerEventMetadataDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<DebuggerEventMetadata>
      _instance;
} _DebuggerEventMetadata_default_instance_;
}  // namespace debug
}  // namespace core
}  // namespace tensorflow
}  // namespace third_party
namespace protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto {
static void InitDefaultsDebuggerEventMetadata() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::third_party::tensorflow::core::debug::_DebuggerEventMetadata_default_instance_;
    new (ptr) ::third_party::tensorflow::core::debug::DebuggerEventMetadata();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::third_party::tensorflow::core::debug::DebuggerEventMetadata::InitAsDefaultInstance();
}

::google::protobuf::internal::SCCInfo<0> scc_info_DebuggerEventMetadata =
    {{ATOMIC_VAR_INIT(::google::protobuf::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsDebuggerEventMetadata}, {}};

void InitDefaults() {
  ::google::protobuf::internal::InitSCC(&scc_info_DebuggerEventMetadata.base);
}

::google::protobuf::Metadata file_level_metadata[1];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::third_party::tensorflow::core::debug::DebuggerEventMetadata, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::third_party::tensorflow::core::debug::DebuggerEventMetadata, device_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::third_party::tensorflow::core::debug::DebuggerEventMetadata, output_slot_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::third_party::tensorflow::core::debug::DebuggerEventMetadata, num_chunks_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::third_party::tensorflow::core::debug::DebuggerEventMetadata, chunk_index_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::third_party::tensorflow::core::debug::DebuggerEventMetadata)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::third_party::tensorflow::core::debug::_DebuggerEventMetadata_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  AssignDescriptors(
      "tensorflow/core/debug/debugger_event_metadata.proto", schemas, file_default_instances, TableStruct::offsets,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 1);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n3tensorflow/core/debug/debugger_event_m"
      "etadata.proto\022!third_party.tensorflow.co"
      "re.debug\"e\n\025DebuggerEventMetadata\022\016\n\006dev"
      "ice\030\001 \001(\t\022\023\n\013output_slot\030\002 \001(\005\022\022\n\nnum_ch"
      "unks\030\003 \001(\005\022\023\n\013chunk_index\030\004 \001(\005b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 199);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensorflow/core/debug/debugger_event_metadata.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static ::google::protobuf::internal::once_flag once;
  ::google::protobuf::internal::call_once(once, AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto
namespace third_party {
namespace tensorflow {
namespace core {
namespace debug {

// ===================================================================

void DebuggerEventMetadata::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int DebuggerEventMetadata::kDeviceFieldNumber;
const int DebuggerEventMetadata::kOutputSlotFieldNumber;
const int DebuggerEventMetadata::kNumChunksFieldNumber;
const int DebuggerEventMetadata::kChunkIndexFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

DebuggerEventMetadata::DebuggerEventMetadata()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  ::google::protobuf::internal::InitSCC(
      &protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto::scc_info_DebuggerEventMetadata.base);
  SharedCtor();
  // @@protoc_insertion_point(constructor:third_party.tensorflow.core.debug.DebuggerEventMetadata)
}
DebuggerEventMetadata::DebuggerEventMetadata(const DebuggerEventMetadata& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  device_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.device().size() > 0) {
    device_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.device_);
  }
  ::memcpy(&output_slot_, &from.output_slot_,
    static_cast<size_t>(reinterpret_cast<char*>(&chunk_index_) -
    reinterpret_cast<char*>(&output_slot_)) + sizeof(chunk_index_));
  // @@protoc_insertion_point(copy_constructor:third_party.tensorflow.core.debug.DebuggerEventMetadata)
}

void DebuggerEventMetadata::SharedCtor() {
  device_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&output_slot_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&chunk_index_) -
      reinterpret_cast<char*>(&output_slot_)) + sizeof(chunk_index_));
}

DebuggerEventMetadata::~DebuggerEventMetadata() {
  // @@protoc_insertion_point(destructor:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  SharedDtor();
}

void DebuggerEventMetadata::SharedDtor() {
  device_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void DebuggerEventMetadata::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const ::google::protobuf::Descriptor* DebuggerEventMetadata::descriptor() {
  ::protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const DebuggerEventMetadata& DebuggerEventMetadata::default_instance() {
  ::google::protobuf::internal::InitSCC(&protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto::scc_info_DebuggerEventMetadata.base);
  return *internal_default_instance();
}


void DebuggerEventMetadata::Clear() {
// @@protoc_insertion_point(message_clear_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  device_.ClearToEmptyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  ::memset(&output_slot_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&chunk_index_) -
      reinterpret_cast<char*>(&output_slot_)) + sizeof(chunk_index_));
  _internal_metadata_.Clear();
}

bool DebuggerEventMetadata::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  for (;;) {
    ::std::pair<::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string device = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(10u /* 10 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_device()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->device().data(), static_cast<int>(this->device().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "third_party.tensorflow.core.debug.DebuggerEventMetadata.device"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 output_slot = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(16u /* 16 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &output_slot_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 num_chunks = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(24u /* 24 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &num_chunks_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 chunk_index = 4;
      case 4: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(32u /* 32 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &chunk_index_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  return false;
#undef DO_
}

void DebuggerEventMetadata::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string device = 1;
  if (this->device().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->device().data(), static_cast<int>(this->device().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "third_party.tensorflow.core.debug.DebuggerEventMetadata.device");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->device(), output);
  }

  // int32 output_slot = 2;
  if (this->output_slot() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(2, this->output_slot(), output);
  }

  // int32 num_chunks = 3;
  if (this->num_chunks() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(3, this->num_chunks(), output);
  }

  // int32 chunk_index = 4;
  if (this->chunk_index() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(4, this->chunk_index(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:third_party.tensorflow.core.debug.DebuggerEventMetadata)
}

::google::protobuf::uint8* DebuggerEventMetadata::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string device = 1;
  if (this->device().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->device().data(), static_cast<int>(this->device().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "third_party.tensorflow.core.debug.DebuggerEventMetadata.device");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        1, this->device(), target);
  }

  // int32 output_slot = 2;
  if (this->output_slot() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(2, this->output_slot(), target);
  }

  // int32 num_chunks = 3;
  if (this->num_chunks() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(3, this->num_chunks(), target);
  }

  // int32 chunk_index = 4;
  if (this->chunk_index() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(4, this->chunk_index(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  return target;
}

size_t DebuggerEventMetadata::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string device = 1;
  if (this->device().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->device());
  }

  // int32 output_slot = 2;
  if (this->output_slot() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->output_slot());
  }

  // int32 num_chunks = 3;
  if (this->num_chunks() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->num_chunks());
  }

  // int32 chunk_index = 4;
  if (this->chunk_index() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int32Size(
        this->chunk_index());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void DebuggerEventMetadata::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  GOOGLE_DCHECK_NE(&from, this);
  const DebuggerEventMetadata* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const DebuggerEventMetadata>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:third_party.tensorflow.core.debug.DebuggerEventMetadata)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:third_party.tensorflow.core.debug.DebuggerEventMetadata)
    MergeFrom(*source);
  }
}

void DebuggerEventMetadata::MergeFrom(const DebuggerEventMetadata& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.device().size() > 0) {

    device_.AssignWithDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.device_);
  }
  if (from.output_slot() != 0) {
    set_output_slot(from.output_slot());
  }
  if (from.num_chunks() != 0) {
    set_num_chunks(from.num_chunks());
  }
  if (from.chunk_index() != 0) {
    set_chunk_index(from.chunk_index());
  }
}

void DebuggerEventMetadata::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void DebuggerEventMetadata::CopyFrom(const DebuggerEventMetadata& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:third_party.tensorflow.core.debug.DebuggerEventMetadata)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool DebuggerEventMetadata::IsInitialized() const {
  return true;
}

void DebuggerEventMetadata::Swap(DebuggerEventMetadata* other) {
  if (other == this) return;
  InternalSwap(other);
}
void DebuggerEventMetadata::InternalSwap(DebuggerEventMetadata* other) {
  using std::swap;
  device_.Swap(&other->device_, &::google::protobuf::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(output_slot_, other->output_slot_);
  swap(num_chunks_, other->num_chunks_);
  swap(chunk_index_, other->chunk_index_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
}

::google::protobuf::Metadata DebuggerEventMetadata::GetMetadata() const {
  protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensorflow_2fcore_2fdebug_2fdebugger_5fevent_5fmetadata_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace debug
}  // namespace core
}  // namespace tensorflow
}  // namespace third_party
namespace google {
namespace protobuf {
template<> GOOGLE_PROTOBUF_ATTRIBUTE_NOINLINE ::third_party::tensorflow::core::debug::DebuggerEventMetadata* Arena::CreateMaybeMessage< ::third_party::tensorflow::core::debug::DebuggerEventMetadata >(Arena* arena) {
  return Arena::CreateInternal< ::third_party::tensorflow::core::debug::DebuggerEventMetadata >(arena);
}
}  // namespace protobuf
}  // namespace google

// @@protoc_insertion_point(global_scope)
