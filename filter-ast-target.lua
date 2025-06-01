-- filter-ast-target.lua
local collected_figure_captions = {}
local collected_table_captions = {}
local figure_elements_seen = 0
local table_elements_seen = 0

-- This filter should run AFTER pandoc-crossref
-- pandoc-crossref will have processed/prefixed the captions

function Figure(fig_block)
  figure_elements_seen = figure_elements_seen + 1
  io.stderr:write("LUA AST Target: Processing Figure block with ID: " .. pandoc.utils.stringify(fig_block.attr.identifier) .. "\n")

  -- fig_block.caption is a pandoc.Caption object.
  -- fig_block.caption.long is a list of Blocks.
  -- pandoc-crossref should have already put "Figure X: " into this caption content.
  if fig_block.caption and fig_block.caption.long and #fig_block.caption.long > 0 then
    local caption_inlines_combined = {}
    for _, block_in_caption in ipairs(fig_block.caption.long) do
      if block_in_caption.content and #block_in_caption.content > 0 then
        if #caption_inlines_combined > 0 then table.insert(caption_inlines_combined, pandoc.Space()) end
        for _, inline_element in ipairs(block_in_caption.content) do
          table.insert(caption_inlines_combined, inline_element)
        end
      end
    end
    if #caption_inlines_combined > 0 then
        table.insert(collected_figure_captions, caption_inlines_combined)
        io.stderr:write("LUA AST Target: Collected figure caption. Attempting to remove Figure block.\n")
    else
        io.stderr:write("LUA AST Target: Figure caption object present but content empty. Attempting to remove Figure block anyway.\n")
    end
  else
    io.stderr:write("LUA AST Target: Figure has no caption or empty caption. Attempting to remove Figure block anyway.\n")
  end
  return {} -- Attempt to remove the entire Figure block
end

function Table(tbl_block)
  table_elements_seen = table_elements_seen + 1
  io.stderr:write("LUA AST Target: Processing Table block with ID: " .. pandoc.utils.stringify(tbl_block.attr.identifier) .. "\n")

  -- tbl_block.caption is a pandoc.Caption object.
  -- tbl_block.caption.long is a list of Blocks.
  -- pandoc-crossref should have already put "Table Y: " into this.
  if tbl_block.caption and tbl_block.caption.long and #tbl_block.caption.long > 0 then
    local caption_inlines_combined = {}
    for _, block_in_caption in ipairs(tbl_block.caption.long) do
      if block_in_caption.content and #block_in_caption.content > 0 then
        if #caption_inlines_combined > 0 then table.insert(caption_inlines_combined, pandoc.Space()) end
        for _, inline_element in ipairs(block_in_caption.content) do
          table.insert(caption_inlines_combined, inline_element)
        end
      end
    end
    if #caption_inlines_combined > 0 then
        table.insert(collected_table_captions, caption_inlines_combined)
        io.stderr:write("LUA AST Target: Collected table caption. Attempting to remove Table block.\n")
    else
        io.stderr:write("LUA AST Target: Table caption object present but content empty. Attempting to remove Table block anyway.\n")
    end
  else
    io.stderr:write("LUA AST Target: Table has no caption. Attempting to remove Table block anyway.\n")
  end
  return {} -- Attempt to remove the entire Table block
end

function Pandoc(doc)
  io.stderr:write("LUA AST Target (Pandoc): Figure elements processed: " .. figure_elements_seen .. "\n")
  io.stderr:write("LUA AST Target (Pandoc): Table elements processed: " .. table_elements_seen .. "\n")
  io.stderr:write("LUA AST Target (Pandoc): Collected figure captions: " .. #collected_figure_captions .. "\n")
  io.stderr:write("LUA AST Target (Pandoc): Collected table captions: " .. #collected_table_captions .. "\n")

  local new_blocks_for_end = {}

  if #collected_figure_captions > 0 then
    table.insert(new_blocks_for_end, pandoc.Header(2, {pandoc.Str("Figure Captions")}))
    for _, caption_inlines in ipairs(collected_figure_captions) do
      table.insert(new_blocks_for_end, pandoc.Para(caption_inlines))
    end
  end

  if #collected_table_captions > 0 then
    table.insert(new_blocks_for_end, pandoc.Header(2, {pandoc.Str("Table Captions")}))
    for _, caption_inlines in ipairs(collected_table_captions) do
      table.insert(new_blocks_for_end, pandoc.Para(caption_inlines))
    end
  end

  for _, block in ipairs(new_blocks_for_end) do
    doc.blocks:insert(block)
  end
  return doc
end
