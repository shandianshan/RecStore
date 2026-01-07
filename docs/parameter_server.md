# RecStore 参数服务器 Parameter Server 模块

## 概述

RecStore 的参数服务器模块是一个分布式参数管理系统，负责在深度学习训练过程中存储、管理和同步模型参数及嵌入表。该模块采用多后端架构设计，支持通过 gRPC 和 bRPC 两种通信协议以及 RDMA 直接内存访问等不同的参数传输机制，能够适应不同的网络环境和性能需求。

PS 模块的核心功能包括参数的获取、更新、初始化、预取等操作。它通过基础层的统一接口定义，使上层应用能够透明地使用不同的后端实现。参数服务器采用了分片管理策略来扩展存储容量，支持多个服务器协同工作以满足大规模参数的存储需求。

## 架构设计

### 模块组织

参数服务器模块遵循分层设计原则，从下至上可以分为四个层次。最底层是基础公共层（base），提供服务器和客户端的抽象接口以及通用的数据结构和工具函数。在此基础上，分别实现了三个具体的协议层：gRPC 层用于基于 HTTP/2 的通用网络通信、bRPC 层用于高性能 RPC 通信、RDMA 层用于低延迟高吞吐的直接内存访问。最上层是统一的 proto 层和入口层，负责协议定义和系统初始化。

整个设计遵循依赖反转原则，具体的实现层都依赖于基础层定义的接口，而不是相反。这种设计使得系统具有良好的可维护性和可扩展性。新的协议实现或通信机制可以通过继承基础接口来无缝集成到系统中。

### 模块关系

PS 模块由五个子模块组成，它们之间的关系如下表所示：

| 模块 | 目录位置 | 主要职责 | 依赖关系 |
|------|---------|---------|---------|
| 基础模块 | `ps/base` | 定义服务器和客户端接口，提供通用工具 | 无内部依赖 |
| gRPC 模块 | `ps/grpc` | 基于 gRPC 的服务器和客户端实现 | 依赖 base、proto |
| bRPC 模块 | `ps/brpc` | 基于 bRPC 的服务器和客户端实现 | 依赖 base、proto |
| RDMA 模块 | `ps/rdma` | 基于 Mayfly 的 RDMA 服务器和客户端 | 依赖 base、mayfly |
| Proto 模块 | `ps/proto` | Protocol Buffer 定义文件 | 无依赖 |

各个协议实现模块（gRPC、bRPC、RDMA）相互独立，可以分别单独编译和部署。在运行时，应用程序通过配置文件选择所需的后端实现。PS 系统的统一入口点 `ps_server.cpp` 根据配置动态加载相应的协议实现。

## 功能详述

### 基础模块（base）

基础模块定义了整个参数服务器系统的核心抽象。它包含两个最重要的类：`BaseParameterServer` 和 `BasePSClient`。

`BaseParameterServer` 是所有服务器实现的抽象基类。它定义了服务器的基本接口，包括 `Init` 方法用于初始化服务器配置和 `Run` 方法用于启动服务器的主循环。具体的协议实现需要继承此类并实现其虚方法，以提供基于特定协议的服务。

`BasePSClient` 是客户端的抽象基类，定义了客户端可以执行的操作集合。这些操作包括 `GetParameter` 用于检索参数值，`PutParameter` 用于初始设置参数值，`UpdateParameter` 用于在训练过程中更新参数。此外还提供了 `InitEmbeddingTable` 用于初始化嵌入表，以及异步操作接口 `AsyncGetParameter` 和 `PrefetchParameter` 用于优化传输效率。

基础模块还定义了关键的数据结构。`EmbeddingTableConfig` 包含了嵌入表的配置信息如嵌入数量和维度。`PSCommand` 是一个枚举类型，定义了系统级的控制命令如清空参数和重加载。

除了接口定义外，基础模块还提供了通用的工具和数据结构。`CachePS` 类实现了参数的本地缓存管理，支持快速访问频繁使用的参数。`Postoffice` 提供了进程间通信的工具函数。`parameters.h` 定义了参数表和嵌入表的通用数据结构。

### gRPC 模块（grpc）

gRPC 模块基于 Google 的 gRPC 框架实现，提供了基于 HTTP/2 的高效通信机制。该模块包含两个核心组件：`ParameterServiceImpl` 服务器类和 `GRPCPSClient` 客户端类。

gRPC 模块的优势在于其广泛的兼容性和成熟的生态。HTTP/2 协议支持多路复用和头部压缩，能够在互联网环境中提供良好的性能。gRPC 还支持多种语言的客户端生成，便于与其他系统集成。

gRPC 的参数传输采用 Protocol Buffer 格式，在 `ps.proto` 中定义。系统定义了多种请求和响应消息类型来支持不同的操作，如 `GetParameterRequest`、`PutParameterRequest` 等。这种消息驱动的设计使得通信协议具有良好的前向兼容性。

分布式的 gRPC 客户端 `DistGRPCPSClient` 通过轮询策略将请求分发到多个参数服务器，实现了水平扩展。它还支持客户端侧的故障重试和超时控制。

### bRPC 模块（brpc）

bRPC 模块是为了满足对超低延迟和高吞吐需求而实现的。bRPC 是由百度开源的 RPC 框架，相比 gRPC 提供了更加优化的通信性能，特别是在局域网和数据中心内的通信场景中。

bRPC 实现包含 `BRPCParameterServiceImpl` 服务器和 `BRPCPSClient` 客户端。与 gRPC 类似，它支持参数的获取、更新和嵌入表的初始化。bRPC 的协议定义在 `ps_brpc.proto` 中。

bRPC 模块的一个关键优势是支持自定义序列化和反序列化逻辑。这使得系统可以针对特定的数据模式进行优化，例如对频繁出现的嵌入向量进行压缩编码。

分布式的 bRPC 客户端 `DistBRPCPSClient` 同样支持多服务器的负载均衡和故障转移。

### RDMA 模块（rdma）

RDMA（Remote Direct Memory Access）模块针对高性能计算场景进行了优化。它基于 Mayfly 分布式共享内存系统构建，利用 RDMA 硬件加速实现了零复制的参数传输。

RDMA 模块实现了 `PETpsServer` 和相关的客户端。相比于网络 RPC，RDMA 直接在网卡级别对远程内存进行访问，绕过了操作系统的网络栈，大大降低了通信延迟。这对于要求极低延迟的场景如在线推理服务是特别有价值的。

Mayfly 系统为 RDMA 通信提供了一层抽象，使得应用程序可以使用统一的接口访问分布式的参数。参数在分布式共享内存中以一致的视图展现，简化了应用的开发复杂度。

### Proto 模块（proto）

Proto 模块包含所有的 Protocol Buffer 定义文件。`ps.proto` 定义了 gRPC 服务的接口和消息类型，`ps_brpc.proto` 定义了 bRPC 服务的接口。这些定义文件是语言无关的，可以自动生成多种编程语言的代码。

Proto 定义将参数操作分为两类：参数表操作和嵌入表操作。参数表操作针对稠密的参数矩阵，而嵌入表操作针对稀疏的嵌入向量。两类操作都支持批量处理来提高效率。

## 工作流程

### 初始化流程

参数服务器的初始化从配置文件开始。应用通过 `recstore_config.json` 指定所使用的 PS 后端、服务器地址、存储容量等参数。根据配置中的 backend 字段，系统在运行时选择相应的 PS 实现。

初始化时，各个 PS 后端通过 `Init` 方法接收配置对象。服务器端解析配置并分配存储资源，客户端获取服务器的地址信息以建立连接。对于分布式场景，多个客户端会连接到多个服务器，形成一个参数服务网络。

### 参数获取流程

在推理或评估阶段，应用调用客户端的 `GetParameter` 方法来检索参数。客户端根据请求的参数键序列构造请求消息，通过网络发送给服务器。服务器查找本地存储的参数值，将结果序列化后返回给客户端。客户端接收响应，反序列化参数值，填充到应用指定的内存缓冲区中。

为了减少延迟，客户端提供了 `AsyncGetParameter` 方法支持异步获取。应用可以在发起请求后继续执行其他任务，稍后检查结果是否就绪。这种方式特别适合于需要获取大量参数但又不能阻塞主线程的场景。

### 参数更新流程

在训练过程中，优化器计算参数的梯度后调用 `UpdateParameter` 方法来更新参数。客户端将梯度信息与参数键一同发送给服务器。服务器接收更新请求，在本地应用优化算法来更新参数值。更新完成后返回确认给客户端。

`PutParameter` 方法用于初始化参数值，通常在训练开始前调用一次。它支持批量设置多个参数的初始值。

### 参数预取流程

为了隐藏网络延迟，PS 系统提供了预取机制。应用调用 `PrefetchParameter` 方法预先获取即将使用的参数，系统返回一个预取 ID。应用可以使用 `IsPrefetchDone` 查询预取是否完成，或使用 `WaitForPrefetch` 同步等待完成。最后通过 `GetPrefetchResult` 获取预取的参数值。

## 文件说明

### 基础模块文件

**base_ps_server.h** 定义了 `BaseParameterServer` 抽象基类。服务器实现必须继承此类并实现 `Run` 虚方法。该文件是整个 PS 系统的服务端接口定义。

**base_client.h** 定义了 `BasePSClient` 抽象基类。客户端实现必须继承此类并实现所有的参数操作方法。该文件定义了客户端可以执行的所有操作。

**cache_ps_impl.h** 实现了参数的本地缓存。`CachePS` 类维护了一个 LRU 缓存，用于存储最近访问过的参数以加速后续访问。这对于工作集较小的应用可以显著提高性能。

**parameters.h** 和 **parameters.cpp** 定义了参数表和嵌入表的数据结构。`Parameters` 类管理稠密参数矩阵的存储和访问，支持行级和列级的分片。嵌入表通过哈希表实现，支持稀疏的参数访问。

**Postoffice.h** 和 **Postoffice.cc** 提供了进程间通信的工具。`Postoffice` 类实现了基于 PSF（Parameter Server Framework）设计的消息分发机制，用于在多进程和多机环境中协调参数服务器的工作。

**config.h** 和 **config.cc** 处理 PS 的配置。`PSConfig` 类从 JSON 配置文件中解析参数服务器的设置，包括后端类型、服务器地址、存储大小等。

**shard_manager.h** 实现了参数分片管理。在分布式部署中，一个参数矩阵可能被分片存储在多个服务器上。`ShardManager` 类维护了分片信息，指导客户端和服务器如何访问分片参数。

### gRPC 模块文件

**grpc_ps_server.cpp** 和 **grpc_ps_server.h** 实现了 gRPC 参数服务器。`ParameterServiceImpl` 继承自 Protocol Buffer 生成的服务基类，实现了所有的 RPC 方法。服务器通过 `grpc::ServerBuilder` 创建，在指定端口监听请求。

**grpc_ps_client.cpp** 和 **grpc_ps_client.h** 实现了 gRPC 客户端。`GRPCPSClient` 继承自 `BasePSClient`，创建与服务器的连接并发送请求。它支持同步和异步操作，以及请求的超时和重试控制。

**dist_grpc_ps_client.cpp** 和 **dist_grpc_ps_client.h** 实现了分布式 gRPC 客户端。`DistGRPCPSClient` 同时连接到多个 gRPC 服务器，实现了参数的分片访问。它根据参数键的哈希值确定参数所在的服务器分片。

**grpc_ps_client_test.cpp** 和 **dist_grpc_ps_client_test.cpp** 包含了 gRPC 客户端的单元测试。

**grpc_ps_client_pytorch.cpp** 为 PyTorch 框架提供了 Python 绑定，使得 Python 应用能够通过 gRPC 与参数服务器通信。

### bRPC 模块文件

**brpc_ps_server.cpp** 和 **brpc_ps_server.h** 实现了 bRPC 参数服务器。`BRPCParameterServiceImpl` 实现了所有的 RPC 方法。相比 gRPC，bRPC 支持更加灵活的控制流和自定义序列化。

**brpc_ps_client.cpp** 和 **brpc_ps_client.h** 实现了 bRPC 客户端。`BRPCPSClient` 使用 bRPC 的 `Channel` 和 `Controller` 与服务器通信。bRPC 的控制器提供了更细粒度的超时和重试控制。

**dist_brpc_ps_client.cpp** 和 **dist_brpc_ps_client.h** 实现了分布式 bRPC 客户端，使用多个客户端实例并行访问不同的服务器分片。

### RDMA 模块文件

**petps_server.cc** 实现了 RDMA 参数服务器 PETps。它基于 Mayfly 分布式共享内存系统，在内存中维护参数表，通过 RDMA 直接提供远程访问。

**petps_client.cc** 和 **petps_client.h** 实现了 RDMA 客户端。`PETpsClient` 通过 Mayfly 的 DSM API 访问远程参数。由于 RDMA 的直接内存访问特性，客户端可以像访问本地内存一样访问远程参数。

**allshards_ps_client.cc** 和 **allshards_ps_client.h** 实现了支持多分片访问的 RDMA 客户端。

**petps_magic.h** 包含了一些特殊的调试和监控接口。

### Proto 模块文件

**ps.proto** 定义了 gRPC 的服务接口和消息类型。主要服务包括 `ParameterService`，定义了 GetParameter、PutParameter、UpdateParameter 等 RPC 方法。消息类型包括各种请求和响应的结构。

**ps_brpc.proto** 定义了 bRPC 的服务接口和消息类型。内容类似于 ps.proto，但针对 bRPC 框架进行了优化。

### 统一入口文件

**ps_server.cpp** 是参数服务器的统一入口点。根据配置文件指定的后端类型，它动态加载并启动相应的 PS 服务器实现（gRPC、bRPC 或 RDMA）。通过工厂模式的设计，应用可以在不修改代码的情况下切换 PS 后端。

**provider.h** 提供了获取客户端实例的工厂接口。应用通过此接口根据配置获取相应后端的客户端对象。

**report_client.h** 提供了性能监控和指标上报的接口。PS 系统在运行过程中收集各项性能指标，如吞吐量、延迟等，定期上报给监控系统。

## 配置说明

参数服务器通过 JSON 格式的配置文件进行配置。可以参考 [./config.md] 进行配置。