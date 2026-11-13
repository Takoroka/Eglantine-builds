// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// This file implements the SSA rewriting algorithm proposed in
//
//      Simple and Efficient Construction of Static Single Assignment Form.
//      Braun M., Buchwald S., Hack S., Leiba R., Mallon C., Zwinkau A. (2013)
//      In: Jhala R., De Bosschere K. (eds)
//      Compiler Construction. CC 2013.
//      Lecture Notes in Computer Science, vol 7791.
//      Springer, Berlin, Heidelberg
//
//      https://link.springer.com/chapter/10.1007/978-3-642-37051-9_6
//

#include <deque>
#include <map>
#include <span>
#include "common/container/unordered_map.h"
#include <variant>
#include <vector>

#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/pred.h"
#include "shader_recompiler/frontend/ir/reg.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
struct FlagTag {
    auto operator<=>(const FlagTag&) const noexcept = default;
};
struct ZeroFlagTag : FlagTag {};
struct SignFlagTag : FlagTag {};
struct CarryFlagTag : FlagTag {};
struct OverflowFlagTag : FlagTag {};

struct GotoVariable : FlagTag {
    GotoVariable() = default;
    explicit GotoVariable(u32 index_) : index{index_} {}

    auto operator<=>(const GotoVariable&) const noexcept = default;

    u32 index;
};

struct IndirectBranchVariable {
    auto operator<=>(const IndirectBranchVariable&) const noexcept = default;
};

using Variant = std::variant<IR::Reg, IR::Pred, ZeroFlagTag, SignFlagTag, CarryFlagTag, OverflowFlagTag, GotoVariable, IndirectBranchVariable>;

// Block::SsaVariableValue indices of the fixed-count variables other than registers.
constexpr size_t ZERO_FLAG_INDEX = IR::NUM_USER_PREDS;
constexpr size_t SIGN_FLAG_INDEX = ZERO_FLAG_INDEX + 1;
constexpr size_t CARRY_FLAG_INDEX = ZERO_FLAG_INDEX + 2;
constexpr size_t OVERFLOW_FLAG_INDEX = ZERO_FLAG_INDEX + 3;
constexpr size_t INDIRECT_BRANCH_INDEX = ZERO_FLAG_INDEX + 4;
static_assert(INDIRECT_BRANCH_INDEX < IR::Block::NUM_SSA_VARIABLES);

struct GotoKey {
    IR::Block* block;
    u32 index;

    bool operator==(const GotoKey&) const noexcept = default;
};

struct GotoKeyHash {
    size_t operator()(const GotoKey& key) const noexcept {
        return std::hash<IR::Block*>{}(key.block) ^
               (static_cast<size_t>(key.index) * 0x9e3779b97f4a7c15ULL);
    }
};

// Registers, predicates, flags and the indirect branch variable are stored in the blocks: the
// reads walk back through blocks and look up a definition in each, which hash maps keyed by
// block made a large part of the pass. Goto variables are not a fixed count.
struct DefTable {
    IR::Value Def(IR::Block* block, IR::Reg variable) {
        return block->SsaRegValue(variable);
    }
    void SetDef(IR::Block* block, IR::Reg variable, const IR::Value& value) {
        block->SetSsaRegValue(variable, value);
    }

    IR::Value Def(IR::Block* block, IR::Pred variable) {
        return block->SsaVariableValue(IR::PredIndex(variable));
    }
    void SetDef(IR::Block* block, IR::Pred variable, const IR::Value& value) {
        block->SetSsaVariableValue(IR::PredIndex(variable), value);
    }

    IR::Value Def(IR::Block* block, GotoVariable variable) {
        const auto it{goto_vars.find(GotoKey{block, variable.index})};
        return it != goto_vars.end() ? it->second : IR::Value{};
    }
    void SetDef(IR::Block* block, GotoVariable variable, const IR::Value& value) {
        goto_vars.insert_or_assign(GotoKey{block, variable.index}, value);
    }

    IR::Value Def(IR::Block* block, IndirectBranchVariable) {
        return block->SsaVariableValue(INDIRECT_BRANCH_INDEX);
    }
    void SetDef(IR::Block* block, IndirectBranchVariable, const IR::Value& value) {
        block->SetSsaVariableValue(INDIRECT_BRANCH_INDEX, value);
    }

    IR::Value Def(IR::Block* block, ZeroFlagTag) {
        return block->SsaVariableValue(ZERO_FLAG_INDEX);
    }
    void SetDef(IR::Block* block, ZeroFlagTag, const IR::Value& value) {
        block->SetSsaVariableValue(ZERO_FLAG_INDEX, value);
    }

    IR::Value Def(IR::Block* block, SignFlagTag) {
        return block->SsaVariableValue(SIGN_FLAG_INDEX);
    }
    void SetDef(IR::Block* block, SignFlagTag, const IR::Value& value) {
        block->SetSsaVariableValue(SIGN_FLAG_INDEX, value);
    }

    IR::Value Def(IR::Block* block, CarryFlagTag) {
        return block->SsaVariableValue(CARRY_FLAG_INDEX);
    }
    void SetDef(IR::Block* block, CarryFlagTag, const IR::Value& value) {
        block->SetSsaVariableValue(CARRY_FLAG_INDEX, value);
    }

    IR::Value Def(IR::Block* block, OverflowFlagTag) {
        return block->SsaVariableValue(OVERFLOW_FLAG_INDEX);
    }
    void SetDef(IR::Block* block, OverflowFlagTag, const IR::Value& value) {
        block->SetSsaVariableValue(OVERFLOW_FLAG_INDEX, value);
    }

    ::Common::unordered_map<GotoKey, IR::Value, GotoKeyHash> goto_vars;
};

IR::Opcode UndefOpcode(IR::Reg) noexcept {
    return IR::Opcode::UndefU32;
}

IR::Opcode UndefOpcode(IR::Pred) noexcept {
    return IR::Opcode::UndefU1;
}

IR::Opcode UndefOpcode(const FlagTag&) noexcept {
    return IR::Opcode::UndefU1;
}

IR::Opcode UndefOpcode(IndirectBranchVariable) noexcept {
    return IR::Opcode::UndefU32;
}

enum class Status {
    Start,
    SetValue,
    PreparePhiArgument,
    PushPhiArgument,
};

struct ReadState {
    ReadState(IR::Block* block_) : block{block_} {}
    ReadState() = default;

    IR::Block* block{};
    IR::Value result{};
    /// The block's phi; null while it is pending, with its operands so far in
    /// Pass::pending_operands (`operand_count` of them from `operands_begin`).
    IR::Inst* phi{};
    size_t operands_begin{};
    size_t operand_count{};
    IR::Block* const* pred_it{};
    IR::Block* const* pred_end{};
    Status pc{Status::Start};
};

class Pass {
public:
    template <typename Type>
    void WriteVariable(Type variable, IR::Block* block, const IR::Value& value) {
        current_def.SetDef(block, variable, value);
    }

    template <typename Type>
    IR::Value ReadVariable(Type variable, IR::Block* root_block) {
        const IR::Opcode undef_opcode{UndefOpcode(variable)};
        // One walk stack for every read (reads never nest)
        stack.clear();
        stack.emplace_back(nullptr);
        stack.emplace_back(root_block);
        const auto prepare_phi_operand{[&] {
            if (stack.back().pred_it == stack.back().pred_end) {
                ReadState& state{stack.back()};
                IR::Block* const block{state.block};
                const IR::Value result{state.phi != nullptr
                                           ? TryRemoveTrivialPhi(*state.phi, block, undef_opcode)
                                           : ResolvePendingPhi(state, undef_opcode)};
                pending_operands.resize(state.operands_begin);
                stack.pop_back();
                stack.back().result = result;
                WriteVariable(variable, block, result);
            } else {
                IR::Block* const imm_pred{*stack.back().pred_it};
                stack.back().pc = Status::PushPhiArgument;
                stack.emplace_back(imm_pred);
            }
        }};
        do {
            IR::Block* const block{stack.back().block};
            switch (stack.back().pc) {
            case Status::Start: {
                if (const IR::Value def = current_def.Def(block, variable); !def.IsEmpty()) {
                    stack.back().result = def;
                } else if (const u32 pending{block->SsaPendingPhi()}; pending != 0) {
                    // A cycle reads the block's pending phi: it is needed after all. Nothing was
                    // inserted at the front of the block meanwhile, so it goes where the
                    // operandless phi used to be created, and gets the same operands.
                    stack.back().result = IR::Value{MaterializePhi(stack[pending], undef_opcode)};
                } else if (!block->IsSsaSealed()) {
                    // Incomplete CFG
                    IR::Inst* phi{&*block->PrependNewInst(block->begin(), IR::Opcode::Phi)};
                    phi->SetFlags(IR::TypeOf(UndefOpcode(variable)));

                    incomplete_phis[block].insert_or_assign(variable, phi);
                    stack.back().result = IR::Value{&*phi};
                } else if (const std::span imm_preds = block->ImmPredecessors();
                           imm_preds.size() == 1) {
                    // Optimize the common case of one predecessor: no phi needed
                    stack.back().pc = Status::SetValue;
                    stack.emplace_back(imm_preds.front());
                    break;
                } else {
                    // The phi stays pending: it is created only when a cycle reads it or when
                    // it merges two different values. Almost all of these are trivial (222,000
                    // of 226,000 in a large title 1 fragment shader), and creating, filling and
                    // turning each one into an identity was most of the pass.
                    ReadState& state{stack.back()};
                    block->SetSsaPendingPhi(static_cast<u32>(stack.size() - 1));
                    state.operands_begin = pending_operands.size();
                    state.pred_it = imm_preds.data();
                    state.pred_end = imm_preds.data() + imm_preds.size();
                    prepare_phi_operand();
                    break;
                }
            }
                [[fallthrough]];
            case Status::SetValue: {
                const IR::Value result{stack.back().result};
                WriteVariable(variable, block, result);
                stack.pop_back();
                stack.back().result = result;
                break;
            }
            case Status::PushPhiArgument: {
                ReadState& state{stack.back()};
                if (state.phi != nullptr) {
                    state.phi->AddPhiOperand(*state.pred_it, state.result);
                } else {
                    pending_operands.emplace_back(*state.pred_it, state.result);
                    ++state.operand_count;
                }
                ++state.pred_it;
            }
                [[fallthrough]];
            case Status::PreparePhiArgument:
                prepare_phi_operand();
                break;
            }
        } while (stack.size() > 1);
        return stack.back().result;
    }

    void SealBlock(IR::Block* block) {
        const auto it{incomplete_phis.find(block)};
        if (it != incomplete_phis.end()) {
            for (auto& pair : it->second) {
                auto& variant{pair.first};
                auto& phi{pair.second};
                std::visit([&](auto& variable) { AddPhiOperands(variable, *phi, block); }, variant);
            }
        }
        block->SsaSeal();
    }

private:
    /// Creates a pending phi with the operands collected so far, at the front of its block.
    IR::Inst* MaterializePhi(ReadState& state, IR::Opcode undef_opcode) {
        IR::Block* const block{state.block};
        IR::Inst* const phi{&*block->PrependNewInst(block->begin(), IR::Opcode::Phi)};
        phi->SetFlags(IR::TypeOf(undef_opcode));
        for (size_t index = 0; index < state.operand_count; ++index) {
            const auto& [pred, value]{pending_operands[state.operands_begin + index]};
            phi->AddPhiOperand(pred, value);
        }
        block->SetSsaPendingPhi(0);
        state.phi = phi;
        return phi;
    }

    /// TryRemoveTrivialPhi for a phi that is still pending (no cycle read it, so it has no
    /// self-references): its value, creating it only when it merges two values. The trivial one
    /// TryRemoveTrivialPhi would have left as an identity without uses leaves nothing.
    IR::Value ResolvePendingPhi(ReadState& state, IR::Opcode undef_opcode) {
        IR::Block* const block{state.block};
        block->SetSsaPendingPhi(0);
        IR::Value same;
        for (size_t index = 0; index < state.operand_count; ++index) {
            const IR::Value& op{pending_operands[state.operands_begin + index].second};
            if (op.Resolve() == same.Resolve()) {
                continue;
            }
            if (!same.IsEmpty()) {
                return IR::Value{MaterializePhi(state, undef_opcode)};
            }
            same = op;
        }
        if (same.IsEmpty()) {
            // Unreachable or the start block: an undefined value after the block's phis
            IR::Block::InstructionList& list{block->Instructions()};
            return IR::Value{
                &*block->PrependNewInst(std::ranges::find_if_not(list, IR::IsPhi), undef_opcode)};
        }
        return same;
    }

    template <typename Type>
    IR::Value AddPhiOperands(Type variable, IR::Inst& phi, IR::Block* block) {
        for (IR::Block* const imm_pred : block->ImmPredecessors()) {
            phi.AddPhiOperand(imm_pred, ReadVariable(variable, imm_pred));
        }
        return TryRemoveTrivialPhi(phi, block, UndefOpcode(variable));
    }

    IR::Value TryRemoveTrivialPhi(IR::Inst& phi, IR::Block* block, IR::Opcode undef_opcode) {
        IR::Value same;
        const size_t num_args{phi.NumArgs()};
        for (size_t arg_index = 0; arg_index < num_args; ++arg_index) {
            const IR::Value& op{phi.Arg(arg_index)};
            if (op.Resolve() == same.Resolve() || op == IR::Value{&phi}) {
                // Unique value or self-reference
                continue;
            }
            if (!same.IsEmpty()) {
                // The phi merges at least two values: not trivial
                return IR::Value{&phi};
            }
            same = op;
        }
        // Remove the phi node from the block, it will be reinserted
        IR::Block::InstructionList& list{block->Instructions()};
        list.erase(IR::Block::InstructionList::s_iterator_to(phi));

        // Find the first non-phi instruction and use it as an insertion point
        IR::Block::iterator reinsert_point{std::ranges::find_if_not(list, IR::IsPhi)};
        if (same.IsEmpty()) {
            // The phi is unreachable or in the start block
            // Insert an undefined instruction and make it the phi node replacement
            // The "phi" node reinsertion point is specified after this instruction
            reinsert_point = block->PrependNewInst(reinsert_point, undef_opcode);
            same = IR::Value{&*reinsert_point};
            ++reinsert_point;
        }
        // Reinsert the phi node and reroute all its uses to the "same" value
        list.insert(reinsert_point, phi);
        phi.ReplaceUsesWith(same);
        // TODO: Try to recursively remove all phi users, which might have become trivial
        return same;
    }

    // TODO: Windows dies with stack exhaustion?
    std::unordered_map<IR::Block*, std::map<Variant, IR::Inst*>> incomplete_phis;
    DefTable current_def;
    std::vector<ReadState> stack;
    std::vector<std::pair<IR::Block*, IR::Value>> pending_operands;
};

void VisitInst(Pass& pass, IR::Block* block, IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::SetRegister:
        if (const IR::Reg reg{inst.Arg(0).Reg()}; reg != IR::Reg::RZ) {
            pass.WriteVariable(reg, block, inst.Arg(1));
        }
        break;
    case IR::Opcode::SetPred:
        if (const IR::Pred pred{inst.Arg(0).Pred()}; pred != IR::Pred::PT) {
            pass.WriteVariable(pred, block, inst.Arg(1));
        }
        break;
    case IR::Opcode::SetGotoVariable:
        pass.WriteVariable(GotoVariable{inst.Arg(0).U32()}, block, inst.Arg(1));
        break;
    case IR::Opcode::SetIndirectBranchVariable:
        pass.WriteVariable(IndirectBranchVariable{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetZFlag:
        pass.WriteVariable(ZeroFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetSFlag:
        pass.WriteVariable(SignFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetCFlag:
        pass.WriteVariable(CarryFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::SetOFlag:
        pass.WriteVariable(OverflowFlagTag{}, block, inst.Arg(0));
        break;
    case IR::Opcode::GetRegister:
        if (const IR::Reg reg{inst.Arg(0).Reg()}; reg != IR::Reg::RZ) {
            inst.ReplaceUsesWith(pass.ReadVariable(reg, block));
        }
        break;
    case IR::Opcode::GetPred:
        if (const IR::Pred pred{inst.Arg(0).Pred()}; pred != IR::Pred::PT) {
            inst.ReplaceUsesWith(pass.ReadVariable(pred, block));
        }
        break;
    case IR::Opcode::GetGotoVariable:
        inst.ReplaceUsesWith(pass.ReadVariable(GotoVariable{inst.Arg(0).U32()}, block));
        break;
    case IR::Opcode::GetIndirectBranchVariable:
        inst.ReplaceUsesWith(pass.ReadVariable(IndirectBranchVariable{}, block));
        break;
    case IR::Opcode::GetZFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(ZeroFlagTag{}, block));
        break;
    case IR::Opcode::GetSFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(SignFlagTag{}, block));
        break;
    case IR::Opcode::GetCFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(CarryFlagTag{}, block));
        break;
    case IR::Opcode::GetOFlag:
        inst.ReplaceUsesWith(pass.ReadVariable(OverflowFlagTag{}, block));
        break;
    default:
        break;
    }
}

void VisitBlock(Pass& pass, IR::Block* block) {
    for (IR::Inst& inst : block->Instructions()) {
        VisitInst(pass, block, inst);
    }
    pass.SealBlock(block);
}

IR::Type GetConcreteType(IR::Inst* inst) {
    std::deque<IR::Inst*> queue;
    queue.push_back(inst);
    while (!queue.empty()) {
        IR::Inst* current = queue.front();
        queue.pop_front();
        const size_t num_args{current->NumArgs()};
        for (size_t i = 0; i < num_args; ++i) {
            const auto set_type = current->Arg(i).Type();
            if (set_type != IR::Type::Opaque) {
                return set_type;
            }
            if (!current->Arg(i).IsImmediate()) {
                queue.push_back(current->Arg(i).Inst());
            }
        }
    }
    return IR::Type::Opaque;
}
} // Anonymous namespace

void SsaRewritePass(IR::Program& program) {
    Pass pass;
    const auto end{program.post_order_blocks.rend()};
    for (auto block = program.post_order_blocks.rbegin(); block != end; ++block) {
        VisitBlock(pass, *block);
    }
    for (auto block = program.post_order_blocks.rbegin(); block != end; ++block) {
        for (IR::Inst& inst : (*block)->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::Phi) {
                if (inst.Type() == IR::Type::Opaque) {
                    inst.SetFlags(GetConcreteType(&inst));
                }
                inst.OrderPhiArgs();
            }
        }
    }
}

} // namespace Shader::Optimization
