-- filter-export-tables-only.lua
local WORD_TEMPLATE_PATH = "word_template.docx"
local table_export_counter = 0

local function sanitize_filename(str)
  if str == nil or str == "" then return "untitled" end
  local sanitized = string.gsub(str, "[^%w_%.%-]", "_")
  sanitized = string.gsub(sanitized, "^[._%-]+", "")
  sanitized = string.gsub(sanitized, "[._%-]+$", "")
  if sanitized == "" then return "sanitized_id" end
  return sanitized
end

-- Function to check if a table has any actual content in its cells
local function table_has_any_cell_content(tbl_node)
  if tbl_node.head and tbl_node.head.rows then
    for _, row in ipairs(tbl_node.head.rows) do
      if row.cells then
        for _, cell in ipairs(row.cells) do
          if cell.contents and #cell.contents > 0 then return true end
        end
      end
    end
  end
  if tbl_node.bodies then
    for _, body_element in ipairs(tbl_node.bodies) do
      if body_element.head then
        for _, row in ipairs(body_element.head) do
          if row.cells then
            for _, cell in ipairs(row.cells) do
              if cell.contents and #cell.contents > 0 then return true end
            end
          end
        end
      end
      if body_element.body then
        for _, row in ipairs(body_element.body) do
          if row.cells then
            for _, cell in ipairs(row.cells) do
              if cell.contents and #cell.contents > 0 then return true end
            end
          end
        end
      end
    end
  end
  if tbl_node.foot and tbl_node.foot.rows then
    for _, row in ipairs(tbl_node.foot.rows) do
      if row.cells then
        for _, cell in ipairs(row.cells) do
          if cell.contents and #cell.contents > 0 then return true end
        end
      end
    end
  end
  return false
end

-- Function to check if a table has a non-empty caption
local function table_has_non_empty_caption(tbl_node)
  if tbl_node.caption and tbl_node.caption.long and #tbl_node.caption.long > 0 then
    -- Further check if any block in the caption has actual content
    for _, block_in_caption in ipairs(tbl_node.caption.long) do
      if block_in_caption.content and #block_in_caption.content > 0 then
        -- Check if content is more than just whitespace (basic check)
        local caption_text = pandoc.utils.stringify(block_in_caption)
        if string.match(caption_text, "%S") then -- Check for any non-whitespace character
            return true
        end
      end
    end
  end
  return false
end


function Table(tbl_block)
  table_export_counter = table_export_counter + 1
  local table_id_str = ""
  if tbl_block.attr and tbl_block.attr.identifier then
    table_id_str = pandoc.utils.stringify(tbl_block.attr.identifier)
  end

  io.stderr:write("------------------------------------------------------------\n")
  io.stderr:write("LUA Table Exporter (Export Only): Invoked. Counter: " .. table_export_counter .. ", ID: '" .. table_id_str .. "'\n")

  local has_content = table_has_any_cell_content(tbl_block)
  local has_caption = table_has_non_empty_caption(tbl_block)

  if table_export_counter == 1 then
    local structure_desc = "First table structure: "
    structure_desc = structure_desc .. "Has actual caption? " .. tostring(has_caption) .. ". "
    structure_desc = structure_desc .. "Has cell content? " .. tostring(has_content) .. ". "
    local head_rows = 0; if tbl_block.head and tbl_block.head.rows then head_rows = #tbl_block.head.rows end
    structure_desc = structure_desc .. "Head rows: " .. head_rows .. ". "
    local body_count = 0; if tbl_block.bodies then body_count = #tbl_block.bodies end
    structure_desc = structure_desc .. "Body count: " .. body_count .. ". "
    if body_count > 0 and tbl_block.bodies[1] then
        local first_body_head_rows = 0; if tbl_block.bodies[1].head then first_body_head_rows = #tbl_block.bodies[1].head end
        local first_body_body_rows = 0; if tbl_block.bodies[1].body then first_body_body_rows = #tbl_block.bodies[1].body end
        structure_desc = structure_desc .. "First body head rows: " .. first_body_head_rows .. ", body rows: " .. first_body_body_rows .. ". "
    end
    local foot_rows = 0; if tbl_block.foot and tbl_block.foot.rows then foot_rows = #tbl_block.foot.rows end
    structure_desc = structure_desc .. "Foot rows: " .. foot_rows .. "."
    io.stderr:write("LUA Table Exporter (Export Only): " .. structure_desc .. "\n")
  end

  -- Check conditions for skipping export
  if not has_caption then
    io.stderr:write("LUA Table Exporter (Export Only): Skipping export for Table (ID: '" .. table_id_str .. "', Counter: " .. table_export_counter .. "). It has no non-empty caption.\n")
    io.stderr:write("------------------------------------------------------------\n")
    return nil
  end
  if not has_content then
    io.stderr:write("LUA Table Exporter (Export Only): Skipping export for Table (ID: '" .. table_id_str .. "', Counter: " .. table_export_counter .. "). It has a caption but no content in its cells.\n")
    io.stderr:write("------------------------------------------------------------\n")
    return nil
  end

  io.stderr:write("LUA Table Exporter (Export Only): Processing valid Table for export (ID: '" .. table_id_str .. "', Counter: " .. table_export_counter .. ") - Has caption and cell content.\n")

  local standalone_doc_blocks = {}
  -- Add the caption blocks (we already know it's non-empty from the check above)
  for _, cap_block in ipairs(tbl_block.caption.long) do
    table.insert(standalone_doc_blocks, cap_block)
  end
  table.insert(standalone_doc_blocks, tbl_block)

  local mini_doc_metadata = pandoc.Meta({})
  local mini_doc = pandoc.Pandoc(standalone_doc_blocks, mini_doc_metadata)

  local json_input_for_pandoc
  local success_serialize, result_serialize = pcall(pandoc.write, mini_doc, 'json')
  if not success_serialize then
    io.stderr:write("LUA Table Exporter (Export Only): ERROR - Failed to serialize mini-document to JSON for table ID '".. table_id_str .."': " .. tostring(result_serialize) .. "\n")
    io.stderr:write("------------------------------------------------------------\n")
    return nil
  else
    json_input_for_pandoc = result_serialize
  end

  local filename_base
  if table_id_str and table_id_str ~= "" then
    filename_base = "table_" .. sanitize_filename(table_id_str)
  else
    filename_base = "table_export_" .. table_export_counter
  end
  local output_filename = filename_base .. ".docx"

  io.stderr:write("LUA Table Exporter (Export Only): Attempting to export '" .. table_id_str .. "' to: " .. output_filename .. "\n")

  local template_file, err_open = io.open(WORD_TEMPLATE_PATH, "r")
  if not template_file then
    io.stderr:write("LUA Table Exporter (Export Only): ERROR - Reference template '" .. WORD_TEMPLATE_PATH .. "' not found. Error: " .. tostring(err_open) .. ". Cannot export " .. output_filename .. "\n")
    io.stderr:write("------------------------------------------------------------\n")
  else
    template_file:close()
    local pandoc_args = { "-f", "json", "-t", "docx", "-o", output_filename, "--reference-doc=" .. WORD_TEMPLATE_PATH }
    local ok_pipe, err_msg_pipe = pandoc.pipe("pandoc", pandoc_args, json_input_for_pandoc)
    if ok_pipe then
      io.stderr:write("LUA Table Exporter (Export Only): Successfully exported to " .. output_filename .. "\n")
    else
      io.stderr:write("LUA Table Exporter (Export Only): ERROR exporting to " .. output_filename .. ". Pandoc pipe error: " .. tostring(err_msg_pipe) .. "\n")
    end
  end
  io.stderr:write("LUA Table Exporter (Export Only): Table (ID: '" .. table_id_str .. "') passed through.\n")
  io.stderr:write("------------------------------------------------------------\n")
  return nil
end