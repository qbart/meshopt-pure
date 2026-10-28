//! jsmn JSON tokenizer (https://github.com/zserge/jsmn), as embedded in cgltf: `JSMN_STRICT` + `JSMN_PARENT_LINKS`.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JsmnType {
    #[default]
    Undefined = 0,
    Object = 1,
    Array = 2,
    String = 3,
    Primitive = 4,
}

/// Not enough tokens were provided
pub const JSMN_ERROR_NOMEM: i32 = -1;
/// Invalid character inside JSON string
pub const JSMN_ERROR_INVAL: i32 = -2;
/// The string is not a full JSON packet, more bytes expected
pub const JSMN_ERROR_PART: i32 = -3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JsmnTok {
    pub type_: JsmnType,
    pub start: isize,
    pub end: isize,
    pub size: i32,
    pub parent: i32,
}

impl Default for JsmnTok {
    fn default() -> Self {
        Self {
            type_: JsmnType::Undefined,
            start: -1,
            end: -1,
            size: 0,
            parent: -1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct JsmnParser {
    /// offset in the JSON string
    pub pos: usize,
    /// next token to allocate
    pub toknext: u32,
    /// superior token node, e.g parent object or array
    pub toksuper: i32,
}

impl JsmnParser {
    /// `jsmn_init`
    pub fn new() -> Self {
        Self {
            pos: 0,
            toknext: 0,
            toksuper: -1,
        }
    }
}

impl Default for JsmnParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Allocates a fresh unused token from the token pool.
fn alloc_token(parser: &mut JsmnParser, tokens: &mut [JsmnTok]) -> Option<usize> {
    let index = parser.toknext as usize;
    let tok = tokens.get_mut(index)?;
    parser.toknext += 1;
    tok.start = -1;
    tok.end = -1;
    tok.size = 0;
    tok.parent = -1;
    Some(index)
}

/// Fills token type and boundaries.
fn fill_token(token: &mut JsmnTok, type_: JsmnType, start: isize, end: isize) {
    token.type_ = type_;
    token.start = start;
    token.end = end;
    token.size = 0;
}

fn byte(js: &[u8], pos: usize) -> u8 {
    js.get(pos).copied().unwrap_or(0)
}

/// Fills next available token with JSON primitive.
fn parse_primitive(parser: &mut JsmnParser, js: &[u8], len: usize, tokens: Option<&mut [JsmnTok]>) -> i32 {
    let start = parser.pos;

    let mut found = false;
    while parser.pos < len && byte(js, parser.pos) != 0 {
        let c = byte(js, parser.pos);
        if matches!(c, b'\t' | b'\r' | b'\n' | b' ' | b',' | b']' | b'}') {
            found = true;
            break;
        }
        // note: C compares a (signed) char; bytes >= 128 are negative there and fail the `< 32` check
        if !(32..127).contains(&c) {
            parser.pos = start;
            return JSMN_ERROR_INVAL;
        }
        parser.pos += 1;
    }

    if !found {
        // In strict mode primitive must be followed by a comma/object/array
        parser.pos = start;
        return JSMN_ERROR_PART;
    }

    let Some(tokens) = tokens else {
        parser.pos -= 1;
        return 0;
    };

    let Some(index) = alloc_token(parser, tokens) else {
        parser.pos = start;
        return JSMN_ERROR_NOMEM;
    };
    let toksuper = parser.toksuper;
    let token = &mut tokens[index];
    fill_token(token, JsmnType::Primitive, start as isize, parser.pos as isize);
    token.parent = toksuper;
    parser.pos -= 1;
    0
}

/// Fills next token with JSON string.
fn parse_string(parser: &mut JsmnParser, js: &[u8], len: usize, tokens: Option<&mut [JsmnTok]>) -> i32 {
    let start = parser.pos;

    parser.pos += 1;

    // Skip starting quote
    while parser.pos < len && byte(js, parser.pos) != 0 {
        let c = byte(js, parser.pos);

        // Quote: end of string
        if c == b'"' {
            let Some(tokens) = tokens else {
                return 0;
            };
            let Some(index) = alloc_token(parser, tokens) else {
                parser.pos = start;
                return JSMN_ERROR_NOMEM;
            };
            let toksuper = parser.toksuper;
            let token = &mut tokens[index];
            fill_token(token, JsmnType::String, start as isize + 1, parser.pos as isize);
            token.parent = toksuper;
            return 0;
        }

        // Backslash: Quoted symbol expected
        if c == b'\\' && parser.pos + 1 < len {
            parser.pos += 1;
            match byte(js, parser.pos) {
                // Allowed escaped symbols
                b'"' | b'/' | b'\\' | b'b' | b'f' | b'r' | b'n' | b't' => {}
                // Allows escaped symbol \uXXXX
                b'u' => {
                    parser.pos += 1;
                    let mut i = 0;
                    while i < 4 && parser.pos < len && byte(js, parser.pos) != 0 {
                        // If it isn't a hex character we have an error
                        if !byte(js, parser.pos).is_ascii_hexdigit() {
                            parser.pos = start;
                            return JSMN_ERROR_INVAL;
                        }
                        parser.pos += 1;
                        i += 1;
                    }
                    parser.pos -= 1;
                }
                // Unexpected symbol
                _ => {
                    parser.pos = start;
                    return JSMN_ERROR_INVAL;
                }
            }
        }

        parser.pos += 1;
    }
    parser.pos = start;
    JSMN_ERROR_PART
}

/// Parse JSON string and fill tokens. `tokens == None` counts the tokens instead.
pub fn jsmn_parse(parser: &mut JsmnParser, js: &[u8], len: usize, mut tokens: Option<&mut [JsmnTok]>) -> i32 {
    let mut count = parser.toknext as i32;

    while parser.pos < len && byte(js, parser.pos) != 0 {
        let c = byte(js, parser.pos);
        match c {
            b'{' | b'[' => {
                count += 1;
                if let Some(tokens) = tokens.as_deref_mut() {
                    let Some(index) = alloc_token(parser, tokens) else {
                        return JSMN_ERROR_NOMEM;
                    };
                    if parser.toksuper != -1 {
                        let toksuper = parser.toksuper as usize;
                        tokens[toksuper].size += 1;
                        tokens[index].parent = parser.toksuper;
                    }
                    let token = &mut tokens[index];
                    token.type_ = if c == b'{' { JsmnType::Object } else { JsmnType::Array };
                    token.start = parser.pos as isize;
                    parser.toksuper = parser.toknext as i32 - 1;
                }
            }
            b'}' | b']' => {
                if let Some(tokens) = tokens.as_deref_mut() {
                    let type_ = if c == b'}' { JsmnType::Object } else { JsmnType::Array };
                    if parser.toknext < 1 {
                        return JSMN_ERROR_INVAL;
                    }
                    let mut index = parser.toknext as usize - 1;
                    loop {
                        let token = tokens[index];
                        if token.start != -1 && token.end == -1 {
                            if token.type_ != type_ {
                                return JSMN_ERROR_INVAL;
                            }
                            tokens[index].end = parser.pos as isize + 1;
                            parser.toksuper = token.parent;
                            break;
                        }
                        if token.parent == -1 {
                            if token.type_ != type_ || parser.toksuper == -1 {
                                return JSMN_ERROR_INVAL;
                            }
                            break;
                        }
                        index = token.parent as usize;
                    }
                }
            }
            b'"' => {
                let r = parse_string(parser, js, len, tokens.as_deref_mut());
                if r < 0 {
                    return r;
                }
                count += 1;
                if parser.toksuper != -1
                    && let Some(tokens) = tokens.as_deref_mut()
                {
                    tokens[parser.toksuper as usize].size += 1;
                }
            }
            b'\t' | b'\r' | b'\n' | b' ' => {}
            b':' => {
                parser.toksuper = parser.toknext as i32 - 1;
            }
            b',' => {
                if let Some(tokens) = tokens.as_deref_mut()
                    && parser.toksuper != -1
                {
                    let t = tokens[parser.toksuper as usize];
                    if t.type_ != JsmnType::Array && t.type_ != JsmnType::Object {
                        parser.toksuper = t.parent;
                    }
                }
            }
            // In strict mode primitives are: numbers and booleans
            b'-' | b'0'..=b'9' | b't' | b'f' | b'n' => {
                // And they must not be keys of the object
                if let Some(tokens) = tokens.as_deref_mut()
                    && parser.toksuper != -1
                {
                    let t = tokens[parser.toksuper as usize];
                    if t.type_ == JsmnType::Object || (t.type_ == JsmnType::String && t.size != 0) {
                        return JSMN_ERROR_INVAL;
                    }
                }
                let r = parse_primitive(parser, js, len, tokens.as_deref_mut());
                if r < 0 {
                    return r;
                }
                count += 1;
                if parser.toksuper != -1
                    && let Some(tokens) = tokens.as_deref_mut()
                {
                    tokens[parser.toksuper as usize].size += 1;
                }
            }
            // Unexpected char in strict mode
            _ => return JSMN_ERROR_INVAL,
        }
        parser.pos += 1;
    }

    if let Some(tokens) = tokens.as_deref() {
        for i in (0..parser.toknext as usize).rev() {
            // Unmatched opened object or array
            if tokens[i].start != -1 && tokens[i].end == -1 {
                return JSMN_ERROR_PART;
            }
        }
    }

    count
}
