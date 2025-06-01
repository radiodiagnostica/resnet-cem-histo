-- filter-ast-target.lua

-- Make sure this function is defined FIRST
local function simple_inspect(val, indent, seen_tables)
  indent = indent or ""
  seen_tables = seen_tables or {}
  local t = type(val)
  if t == "string" then
    return "\"" .. val .. "\""
  elseif t == "number" or t == "boolean" then
    return tostring(val)
  elseif t == "nil" then
    return "nil"
  elseif t == "table" then
    if seen_tables[val] then return "{recursive_ref}" end
    seen_tables[val] = true
    local parts = {}
    local is_sequence = true
    local max_i = 0
    for k, _ in pairs(val) do
      if type(k) ~= "number" or k < 1 or math.floor(k) ~= k then
        is_sequence = false
        break
      end
      if k > max_i then max_i = k end
    end
    if max_i > 0 then -- Check for holes if using #val
        local current_length = 0
        for i=1,max_i do if val[i] ~= nil then current_length = current_length + 1 end end
        if current_length ~= max_i then is_sequence = false end
    elseif #val == 0 and max_i == 0 then -- empty table could be sequence or map
        -- default to map unless explicitly an empty sequence from pandoc AST (which is rare here)
    else -- #val might be 0 if sparse, or if it's a map
        is_sequence = false
    end


    if is_sequence and max_i > 0 then -- Treat as array
        for i = 1, max_i do
             -- Handle nil values in sequence explicitly for inspect
            table.insert(parts, simple_inspect(val[i], indent .. "  ", seen_tables))
        end
        if #parts == 0 then return "{}" end -- for case like {nil,nil} if max_i was >0
        return "{\n" .. indent .. "  " .. table.concat(parts, ",\n" .. indent .. "  ") .. "\n" .. indent .. "}"
    else -- Treat as map
        for k, v in pairs(val) do
            local key_str = type(k) == "string" and k or "["..tostring(k).."]"
            table.insert(parts, key_str .. " = " .. simple_inspect(v, indent .. "  ", seen_tables))
        end
        if #parts == 0 then return "{}" end
        return "{\n" .. indent .. "  " .. table.concat(parts, ",\n" .. indent .. "  ") .. "\n" .. indent .. "}"
    end
  else
    return t .. ": " .. tostring(val)
  end
end

-- THEN define get_meta_string_value using the latest version I provided
local function get_meta_string_value(meta_value, key_name_for_logging)
  if not meta_value then
    io.stderr:write("LUA AST Target (get_meta_string_value): Key '" .. key_name_for_logging .. "' not found in meta, meta_value is nil.\n")
    return nil
  end

  io.stderr:write("LUA DEBUG (get_meta_string_value): For key '" .. key_name_for_logging .. "', meta_value raw type: " .. type(meta_value) .. "\n")
  io.stderr:write("LUA DEBUG (get_meta_string_value): For key '" .. key_name_for_logging .. "', meta_value inspected: " .. simple_inspect(meta_value) .. "\n")

  local meta_value_t = meta_value and meta_value.t
  io.stderr:write("LUA AST Target (get_meta_string_value): Processing key '" .. key_name_for_logging .. "'. Actual MetaValue.t: " .. (meta_value_t or "nil or not present") .. "\n")

  local extracted_str = ""

  if meta_value_t == "MetaInlines" then
    extracted_str = pandoc.utils.stringify(meta_value)
  elseif meta_value_t == "MetaString" then
    extracted_str = meta_value.text
  elseif meta_value_t == "MetaList" and type(meta_value) == "table" and #meta_value > 0 then
    local first_item = meta_value[1]
    if first_item and first_item.t == "MetaInlines" then
      extracted_str = pandoc.utils.stringify(first_item)
    elseif first_item and first_item.t == "MetaString" then
      extracted_str = first_item.text
    else
      io.stderr:write("LUA AST Target (get_meta_string_value): Key '" .. key_name_for_logging .. "' is a MetaList, but first item is nil, or not MetaInlines/MetaString. First item type: " .. (first_item and first_item.t or "nil") .. "\n")
    end
  elseif type(meta_value) == "string" then
    io.stderr:write("LUA AST Target (get_meta_string_value): Key '" .. key_name_for_logging .. "' IS A PLAIN LUA STRING. Value: '" .. meta_value .. "'\n")
    extracted_str = meta_value
  elseif type(meta_value) == "boolean" then
    io.stderr:write("LUA AST Target (get_meta_string_value): Key '" .. key_name_for_logging .. "' IS A PLAIN LUA BOOLEAN. Value: " .. tostring(meta_value) .. "\n")
    extracted_str = tostring(meta_value)
  elseif type(meta_value) == "table" and not meta_value_t then
    io.stderr:write("LUA AST Target (get_meta_string_value): Key '" .. key_name_for_logging .. "' is a table without a '.t' field. Assuming custom structure.\n")
    local found_str_in_custom_table = false
    for key_in_table, val_in_table in pairs(meta_value) do
      if val_in_table then
        local success, str_or_err = pcall(pandoc.utils.stringify, val_in_table)
        if success and type(str_or_err) == "string" and str_or_err ~= "" then
          io.stderr:write("LUA AST Target (get_meta_string_value): Extracted from custom table structure (key: " .. simple_inspect(key_in_table) .. "): '" .. str_or_err .. "'\n")
          extracted_str = str_or_err
          found_str_in_custom_table = true
          break
        else
           io.stderr:write("LUA AST Target (get_meta_string_value): Could not stringify value for key '" .. simple_inspect(key_in_table) .. "' in custom table or it was empty. Error/Val: " .. tostring(str_or_err) .. "\n")
        end
      end
    end
    if not found_str_in_custom_table then
        io.stderr:write("LUA AST Target (get_meta_string_value): Iterated custom table for key '" .. key_name_for_logging .. "' but found no stringifiable values.\n")
    end
  else
    io.stderr:write("LUA AST Target (get_meta_string_value): Key '" .. key_name_for_logging .. "' found but is not MetaInlines, MetaString, MetaList, a plain Lua string/boolean, or recognized custom table. MetaValue.t: " .. (meta_value_t or "nil or not present") .. "\n")
  end

  if extracted_str and extracted_str ~= "" then
    local processed_str = string.gsub(extracted_str, "^%s*(.-)%s*$", "%1")
    if key_name_for_logging == "table_export_format" then
        processed_str = string.lower(processed_str)
    end
    io.stderr:write("LUA AST Target (get_meta_string_value): Extracted and processed string for '" .. key_name_for_logging .. "': '" .. processed_str .. "' from original value: '" .. extracted_str .. "'\n")
    return processed_str
  else
    io.stderr:write("LUA AST Target (get_meta_string_value): Could not extract a usable string for key '" .. key_name_for_logging .. "'. Original value might have been empty or unprocessable.\n")
    return nil
  end
end


-- Global variables for the filter (these are fine)
local collected_figure_captions = {}
local figure_elements_seen = 0
local table_elements_seen = 0
local table_file_counter = 0

local TABLE_EXPORT_FORMAT = 'gfm'
local TABLE_EXPORT_EXTENSION = '.md'
local TABLE_EXPORT_WRITER_OPTIONS = {}

-- Helper function to sanitize strings for use in filenames (this is fine)
local function sanitize_filename(str)
  if not str or str == '' then
    return 'default_name'
  end
  str = string.lower(str)
  str = string.gsub(str, '%s+', '_')
  str = string.gsub(str, '[^%w%-_%.]+', '')
  if str == "." or str == ".." then return "_" .. str .. "_" end
  if str == "" then return "unnamed_table" end
  return str
end

-- Then your Meta, Figure, Table, Pandoc functions using the LATEST versions.
-- For brevity, I'll just show the Meta function that was last refined:
function Meta(meta)
  TABLE_EXPORT_FORMAT = 'gfm'
  TABLE_EXPORT_EXTENSION = '.md'
  TABLE_EXPORT_WRITER_OPTIONS = {}

  io.stderr:write("LUA AST Target (Meta): Initial TABLE_EXPORT_FORMAT: '" .. TABLE_EXPORT_FORMAT .. "'\n")

  if meta.table_export_format then
    local format_str = get_meta_string_value(meta.table_export_format, "table_export_format")
    if format_str then
      TABLE_EXPORT_FORMAT = format_str
    else
      io.stderr:write("LUA AST Target (Meta): 'table_export_format' key existed, but could not extract a usable string. Using default: '" .. TABLE_EXPORT_FORMAT .. "'\n")
    end
  else
    io.stderr:write("LUA AST Target (Meta): No 'table_export_format' key in YAML. Using default: '" .. TABLE_EXPORT_FORMAT .. "'\n")
  end

  if TABLE_EXPORT_FORMAT == "docx" then
    TABLE_EXPORT_EXTENSION = ".docx"
  elseif TABLE_EXPORT_FORMAT == "odt" then
    TABLE_EXPORT_EXTENSION = ".odt"
  elseif TABLE_EXPORT_FORMAT == "html" or TABLE_EXPORT_FORMAT == "htm" then
    TABLE_EXPORT_EXTENSION = ".html"
  elseif TABLE_EXPORT_FORMAT == "latex" or TABLE_EXPORT_FORMAT == "tex" then
    TABLE_EXPORT_EXTENSION = ".tex"
  elseif TABLE_EXPORT_FORMAT == "rst" then
    TABLE_EXPORT_EXTENSION = ".rst"
  elseif TABLE_EXPORT_FORMAT == "pdf" then
    TABLE_EXPORT_EXTENSION = ".pdf"
  elseif TABLE_EXPORT_FORMAT == "gfm" or
         TABLE_EXPORT_FORMAT == "commonmark" or
         TABLE_EXPORT_FORMAT == "markdown" or
         TABLE_EXPORT_FORMAT == "markdown_strict" or
         TABLE_EXPORT_FORMAT == "md" then
    TABLE_EXPORT_EXTENSION = ".md"
  else
    io.stderr:write("LUA AST Target (Meta): WARNING: Unrecognized TABLE_EXPORT_FORMAT '"..TABLE_EXPORT_FORMAT.."' for extension assignment. Defaulting extension to '.out'.\n")
    TABLE_EXPORT_EXTENSION = ".out"
  end
  io.stderr:write("LUA AST Target (Meta): Final determined TABLE_EXPORT_FORMAT: '" .. TABLE_EXPORT_FORMAT .. "', TABLE_EXPORT_EXTENSION: '" .. TABLE_EXPORT_EXTENSION .. "'\n")

  if meta.table_export_reference_doc then
    local ref_doc_path = get_meta_string_value(meta.table_export_reference_doc, "table_export_reference_doc")
    if ref_doc_path then
      if TABLE_EXPORT_FORMAT == "docx" or TABLE_EXPORT_FORMAT == "odt" or TABLE_EXPORT_FORMAT == "pptx" then
        TABLE_EXPORT_WRITER_OPTIONS.reference_doc = ref_doc_path
        io.stderr:write("LUA AST Target (Meta): Using table export reference_doc: '" .. ref_doc_path .. "'\n")
      else
        io.stderr:write("LUA AST Target (Meta): WARNING: 'table_export_reference_doc' ('"..ref_doc_path.."') is specified, but current export format ('"..TABLE_EXPORT_FORMAT.."') is not DOCX, ODT, or PPTX. Ignoring reference_doc.\n")
      end
    else
      io.stderr:write("LUA AST Target (Meta): 'table_export_reference_doc' key existed, but could not extract a usable string.\n")
    end
  else
    io.stderr:write("LUA AST Target (Meta): No 'table_export_reference_doc' key in YAML.\n")
  end
end

-- ... and then the Figure, Table, and Pandoc functions as before.
-- Ensure these are the latest versions you have that were working,
-- or the ones I provided in the response where we finalized get_meta_string_value.
-- (The Table function in particular relies on the global TABLE_EXPORT_FORMAT etc.)

function Figure(fig_block)
  figure_elements_seen = figure_elements_seen + 1
  local fig_id_str = fig_block.attr.identifier and pandoc.utils.stringify(fig_block.attr.identifier) or "N/A"
  io.stderr:write("LUA AST Target (Figure): Processing Figure block with ID: " .. fig_id_str .. "\n")

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
        io.stderr:write("LUA AST Target (Figure): Collected figure caption. Removing Figure block.\n")
    else
        io.stderr:write("LUA AST Target (Figure): Figure caption object present but content empty. Removing Figure block.\n")
    end
  else
    io.stderr:write("LUA AST Target (Figure): Figure has no caption or empty caption. Removing Figure block.\n")
  end
  return {}
end

function Table(tbl_block)
  table_elements_seen = table_elements_seen + 1
  local tbl_id_str = tbl_block.attr.identifier and pandoc.utils.stringify(tbl_block.attr.identifier) or ""

  local filename_base
  if tbl_id_str ~= "" then
    filename_base = "table_" .. sanitize_filename(tbl_id_str)
  else
    table_file_counter = table_file_counter + 1
    filename_base = "table_export_" .. table_file_counter
    tbl_id_str = "generated-id-" .. table_file_counter
    io.stderr:write("LUA AST Target (Table): Table has no ID, assigning temporary for logging: " .. tbl_id_str .. "\n")
  end
  local output_filename = filename_base .. TABLE_EXPORT_EXTENSION

  io.stderr:write("LUA AST Target (Table): Processing Table block with ID '" .. tbl_id_str .. "'. Exporting to: '" .. output_filename .. "' (Format: '" .. TABLE_EXPORT_FORMAT .. "')\n")

  local single_table_doc = pandoc.Pandoc({tbl_block})
  local table_content_string

  local success, result_or_error = pcall(pandoc.write, single_table_doc, TABLE_EXPORT_FORMAT, TABLE_EXPORT_WRITER_OPTIONS)

  if success then
    table_content_string = result_or_error
    if table_content_string == nil then
      io.stderr:write("LUA AST Target (Table): ERROR rendering table '" .. tbl_id_str .. "' to format '" .. TABLE_EXPORT_FORMAT .. "': pandoc.write returned nil.\n")
      return tbl_block
    end
    if type(table_content_string) ~= "string" then
        io.stderr:write("LUA AST Target (Table): ERROR rendering table '" .. tbl_id_str .. "' to format '" .. TABLE_EXPORT_FORMAT .. "': pandoc.write did not return a string (type: " .. type(table_content_string) .. ").\n")
        return tbl_block
    end
  else
    io.stderr:write("LUA AST Target (Table): ERROR rendering table '" .. tbl_id_str .. "' to format '" .. TABLE_EXPORT_FORMAT .. "': " .. tostring(result_or_error) .. "\n")
    return tbl_block
  end

  local file_open_mode = "w"
  local binary_output_formats = { "docx", "odt", "epub", "pdf", "pptx" }
  for _, fmt in ipairs(binary_output_formats) do
    if TABLE_EXPORT_FORMAT == fmt then
      file_open_mode = "wb"
      break
    end
  end
  io.stderr:write("LUA AST Target (Table): File open mode for '" .. output_filename .. "' will be: '" .. file_open_mode .. "'\n")

  local file, ferr = io.open(output_filename, file_open_mode)
  if not file then
    io.stderr:write("LUA AST Target (Table): ERROR opening file '" .. output_filename .. "' for writing (mode " .. file_open_mode .. "): " .. tostring(ferr) .. "\n")
    return tbl_block
  end
  local write_success, write_err = file:write(table_content_string)
  if not write_success then
      io.stderr:write("LUA AST Target (Table): ERROR writing to file '" .. output_filename .. "': " .. tostring(write_err) .. "\n")
      file:close()
      return tbl_block
  end
  file:close()
  io.stderr:write("LUA AST Target (Table): Successfully wrote table '" .. tbl_id_str .. "' to '" .. output_filename .. "'\n")

  local link_text_inlines = {}
  if tbl_block.caption and tbl_block.caption.long and #tbl_block.caption.long > 0 then
    for _, block_in_caption in ipairs(tbl_block.caption.long) do
      if block_in_caption.content and #block_in_caption.content > 0 then
        if #link_text_inlines > 0 then table.insert(link_text_inlines, pandoc.Space()) end
        for _, inline_element in ipairs(block_in_caption.content) do
          table.insert(link_text_inlines, inline_element)
        end
      end
    end
  end

  if #link_text_inlines == 0 then
    local default_text_str = "Table"
    if tbl_id_str ~= "" and not tbl_id_str:match("^generated%-id%-") then
        default_text_str = default_text_str .. " " .. sanitize_filename(tbl_id_str)
    end
    table.insert(link_text_inlines, pandoc.Str(default_text_str .. " (see file " .. output_filename .. ")"))
    io.stderr:write("LUA AST Target (Table): Table '" .. tbl_id_str .. "' has no caption, using default link text.\n")
  else
    io.stderr:write("LUA AST Target (Table): Using existing caption as link text for table '" .. tbl_id_str .. "'.\n")
  end

  return {pandoc.Para({pandoc.Link(link_text_inlines, output_filename)})}
end

function Pandoc(doc)
  io.stderr:write("LUA AST Target (Pandoc): Figure elements processed: " .. figure_elements_seen .. "\n")
  io.stderr:write("LUA AST Target (Pandoc): Table elements processed: " .. table_elements_seen .. "\n")
  io.stderr:write("LUA AST Target (Pandoc): Collected figure captions: " .. #collected_figure_captions .. "\n")

  local new_blocks_for_end = {}

  if #collected_figure_captions > 0 then
    table.insert(new_blocks_for_end, pandoc.Header(2, {pandoc.Str("List of Figures")}))
    local items = {}
    for _, caption_inlines in ipairs(collected_figure_captions) do
      table.insert(items, {pandoc.Para(caption_inlines)})
    end
    table.insert(new_blocks_for_end, pandoc.BulletList(items))
  end

  for _, block in ipairs(new_blocks_for_end) do
    doc.blocks:insert(block)
  end
  return doc
end