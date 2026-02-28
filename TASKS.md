# All-to-Pipe - Implementation Tasks

## Active Tasks

### 1. Replace Hardcoded Constants with ComfyUI Source
**File**: `common/constants.py`
**Priority**: HIGH
**Status**: TODO

**Description**: Currently SUPPORTED_SAMPLERS and SUPPORTED_SCHEDULERS are hardcoded lists. Should import from ComfyUI's actual source instead.

**Tasks**:
- [ ] 1.1: Replace `SUPPORTED_SAMPLERS` hardcoded list with import from `comfy.samplers.SAMPLER_NAMES`
- [ ] 1.2: Replace `SUPPORTED_SCHEDULERS` hardcoded list with import from `comfy.samplers.SCHEDULER_NAMES`
- [ ] 1.3: Verify both lists match expected values after replacement
- [ ] 1.4: Update any code that relies on these constants to handle dynamic imports

**Why**: Hardcoded lists become outdated when ComfyUI updates. Using the source of truth ensures compatibility.

---

### 2. Fix Template Placeholder Syntax to Avoid Conflicts
**File**: `common/prompt_template.py`
**Priority**: CRITICAL
**Status**: TODO

**Description**: Current placeholder syntax uses `{variable}` which conflicts with comfyui-dynamic-prompts module that uses `{}`, `__term__`, `$`, `()`, `|` and other symbols.

**Tasks**:
- [ ] 2.1: Choose alternative placeholder syntax (suggested: `<variable>` or `[[variable]]` or `@variable`)
- [ ] 2.2: Update PLACEHOLDER_PATTERN regex with new syntax
- [ ] 2.3: Update TemplateParser methods to handle new syntax
- [ ] 2.4: Update documentation (COMPANION_FILES.md, docstrings) with new syntax
- [ ] 2.5: Update all test cases to use new syntax
- [ ] 2.6: Verify no conflicts with comfyui-dynamic-prompts

**Recommendation**: Use `<variable>` syntax (single character delimiters, simple, clear)

---

### 3. Reconsider PipeNode Necessity
**File**: `nodes/pipe_node.py`
**Priority**: MEDIUM
**Status**: TODO

**Description**: Current design makes PipeNode unnecessary. All nodes should create a new pipe when no pipe is passed as input. This enables flexibility and removes redundant node.

**Tasks**:
- [ ] 3.1: Evaluate: Should all nodes have optional pipe input?
- [ ] 3.2: If yes: Update all node execute() methods to create Pipe if None
- [ ] 3.3: Update INPUT_TYPES for all nodes to make pipe optional
- [ ] 3.4: Remove PipeNode from nodes list (or keep as convenience node)
- [ ] 3.5: Remove PipeNode from `__init__.py` registration
- [ ] 3.6: Update tests accordingly

**Decision Required**: Keep PipeNode as convenience entry point OR remove completely?

---

### 4. Move Template Parsing to Export Nodes Only
**File**: `nodes/template_parser_node.py`
**Priority**: MEDIUM
**Status**: TODO

**Description**: Template parsing is a data transformation that should happen at export time (when resolving to actual values), not as a separate pipeline node. TemplateParserNode is unnecessary.

**Tasks**:
- [ ] 4.1: Add template parsing logic to `ExportNode.execute()`
- [ ] 4.2: Add template parsing logic to `ExportJsonNode.execute()`
- [ ] 4.3: Remove TemplateParserNode class completely
- [ ] 4.4: Remove TemplateParserNode from nodes list
- [ ] 4.5: Remove TemplateParserNode from `__init__.py` registration
- [ ] 4.6: Update documentation removing TemplateParserNode reference
- [ ] 4.7: Update test suite

**Benefit**: Simpler pipeline, template resolution happens at natural point (export)

---

### 5. Consolidate ExportJsonNode Output Structure
**File**: `nodes/export_json_node.py`
**Priority**: HIGH
**Status**: TODO

**Description**: Current output has multiple nested dictionaries (model, loras, parameters, image_config, positive_prompt, negative_prompt, companion_data). Should consolidate into single flat dictionary for clarity.

**Tasks**:
- [ ] 5.1: Design new consolidated JSON structure
  - Option A: Flat structure with prefixes (model_name, model_subfolder, param_steps, param_cfg, etc)
  - Option B: Single level nested (pipe_data containing all sub-objects)
  - Option C: Hierarchical but cleaner (data > model/loras/parameters/prompts/config)
- [ ] 5.2: Update ExportJsonNode.execute() to produce new structure
- [ ] 5.3: Update COMPANION_FILES.md JSON export examples
- [ ] 5.4: Update tests to match new structure

**Current Structure** (should change):
```json
{
  "model": {...},
  "loras": [...],
  "parameters": {...},
  "image_config": {...},
  "positive_prompt": {...},
  "negative_prompt": {...},
  "companion_data": {...}
}
```

**Suggested New Structure**:
```json
{
  "pipeline": {
    "model": {...},
    "loras": [...],
    "parameters": {...},
    "prompts": {
      "positive": {...},
      "negative": {...}
    },
    "image_config": {...},
    "metadata": {
      "companion": {...}
    }
  }
}
```

---

### 6. Update Node Test Files
**Files**: `nodes/tests/test_*.py`
**Priority**: HIGH
**Status**: TODO

**Description**: Test files need updates to reflect changes to node signatures and removed nodes.

**Tasks**:
- [ ] 6.1: If PipeNode removed: Delete `nodes/tests/test_pipe_node.py`
- [ ] 6.2: If TemplateParserNode removed: Delete `nodes/tests/test_template_parser_node.py`
- [ ] 6.3: Update all node tests to reflect optional pipe inputs
- [ ] 6.4: Update ExportNode tests for template parsing functionality
- [ ] 6.5: Update ExportJsonNode tests for new output structure

---

### 7. Update Constant Usage Throughout Codebase
**Files**: All files using SUPPORTED_SAMPLERS or SUPPORTED_SCHEDULERS
**Priority**: HIGH
**Status**: TODO

**Description**: After importing from ComfyUI, verify all usages still work correctly.

**Files to check**:
- [ ] 7.1: `nodes/parameters_builder_node.py` - Uses SUPPORTED_SAMPLERS in INPUT_TYPES
- [ ] 7.2: All validator code using these constants
- [ ] 7.3: Any error messages referencing these lists

---

### 8. Update Template Documentation
**Files**: `COMPANION_FILES.md`, `ARCHITECTURE.md`, docstrings
**Priority**: MEDIUM
**Status**: TODO

**Description**: Update all documentation to reflect new placeholder syntax and removed nodes.

**Tasks**:
- [ ] 8.1: Update COMPANION_FILES.md examples with new placeholder syntax
- [ ] 8.2: Update ARCHITECTURE.md node diagram (remove PipeNode, TemplateParserNode)
- [ ] 8.3: Update all docstrings mentioning template syntax
- [ ] 8.4: Update README examples
- [ ] 8.5: Update inline comments in code

---

### 9. Verify Module Integration After Changes
**File**: `__init__.py`
**Priority**: HIGH
**Status**: TODO

**Description**: After removing/modifying nodes, ensure module still loads correctly.

**Tasks**:
- [ ] 9.1: Remove removed nodes from NODE_CLASS_MAPPINGS
- [ ] 9.2: Remove removed nodes from NODE_DISPLAY_NAME_MAPPINGS
- [ ] 9.3: Run module load test: `python test_module_load.py`
- [ ] 9.4: Verify CUSTOM_TYPE_NAMES still set correctly
- [ ] 9.5: Run ComfyUI verbose to check no errors

---

## Summary

| Task | Files Affected | Priority | Impact |
|------|----------------|----------|--------|
| Replace hardcoded constants | constants.py | HIGH | Module reliability |
| Fix template syntax | prompt_template.py | CRITICAL | Feature compatibility |
| Reconsider PipeNode | pipe_node.py | MEDIUM | Pipeline design |
| Move template parsing | template_parser_node.py | MEDIUM | Code organization |
| Consolidate JSON output | export_json_node.py | HIGH | API clarity |
| Update tests | nodes/tests/* | HIGH | Code quality |
| Update constants usage | Multiple files | HIGH | Integration |
| Update documentation | Docs + docstrings | MEDIUM | User clarity |
| Module verification | __init__.py | HIGH | Functionality |

