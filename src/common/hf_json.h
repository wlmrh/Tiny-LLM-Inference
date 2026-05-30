#pragma once

#include <cctype>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_llm {
namespace hf_json {

enum class ValueType {
    kNull,
    kBool,
    kNumber,
    kString,
    kArray,
    kObject,
};

struct Value {
    ValueType type = ValueType::kNull;
    bool bool_value = false;
    double number_value = 0.0;
    std::string string_value;
    std::vector<Value> array_value;
    std::unordered_map<std::string, std::unique_ptr<Value>> object_value;

    Value() = default;

    Value(const Value& other)
        : type(other.type),
          bool_value(other.bool_value),
          number_value(other.number_value),
          string_value(other.string_value),
          array_value(other.array_value)
    {
        object_value.reserve(other.object_value.size());
        for (const auto& item : other.object_value)
        {
            if (item.second)
            {
                object_value.emplace(item.first, std::make_unique<Value>(*item.second));
            }
            else
            {
                object_value.emplace(item.first, nullptr);
            }
        }
    }

    Value& operator=(const Value& other)
    {
        if (this == &other)
        {
            return *this;
        }

        type = other.type;
        bool_value = other.bool_value;
        number_value = other.number_value;
        string_value = other.string_value;
        array_value = other.array_value;
        object_value.clear();
        object_value.reserve(other.object_value.size());
        for (const auto& item : other.object_value)
        {
            if (item.second)
            {
                object_value.emplace(item.first, std::make_unique<Value>(*item.second));
            }
            else
            {
                object_value.emplace(item.first, nullptr);
            }
        }

        return *this;
    }

    Value(Value&&) noexcept = default;
    Value& operator=(Value&&) noexcept = default;
    ~Value() = default;

    static Value make_null()
    {
        return Value{};
    }

    static Value make_bool(bool value)
    {
        Value out;
        out.type = ValueType::kBool;
        out.bool_value = value;
        return out;
    }

    static Value make_number(double value)
    {
        Value out;
        out.type = ValueType::kNumber;
        out.number_value = value;
        return out;
    }

    static Value make_string(std::string value)
    {
        Value out;
        out.type = ValueType::kString;
        out.string_value = std::move(value);
        return out;
    }

    static Value make_array(std::vector<Value> value)
    {
        Value out;
        out.type = ValueType::kArray;
        out.array_value = std::move(value);
        return out;
    }

    static Value make_object(std::unordered_map<std::string, std::unique_ptr<Value>> value)
    {
        Value out;
        out.type = ValueType::kObject;
        out.object_value = std::move(value);
        return out;
    }

    bool is_object() const { return type == ValueType::kObject; }
    bool is_array() const { return type == ValueType::kArray; }
    bool is_string() const { return type == ValueType::kString; }
    bool is_number() const { return type == ValueType::kNumber; }

    const std::unordered_map<std::string, std::unique_ptr<Value>>& as_object(const std::string& error_prefix) const
    {
        if (!is_object())
        {
            throw std::runtime_error(error_prefix + ": expected object.");
        }
        return object_value;
    }

    const std::vector<Value>& as_array(const std::string& error_prefix) const
    {
        if (!is_array())
        {
            throw std::runtime_error(error_prefix + ": expected array.");
        }
        return array_value;
    }

    const std::string& as_string(const std::string& error_prefix) const
    {
        if (!is_string())
        {
            throw std::runtime_error(error_prefix + ": expected string.");
        }
        return string_value;
    }

    double as_number(const std::string& error_prefix) const
    {
        if (!is_number())
        {
            throw std::runtime_error(error_prefix + ": expected number.");
        }
        return number_value;
    }

    int64_t as_int64(const std::string& error_prefix) const
    {
        const double value = as_number(error_prefix);
        if (!std::isfinite(value) || std::floor(value) != value)
        {
            throw std::runtime_error(error_prefix + ": expected integer number.");
        }

        if (value < static_cast<double>(INT64_MIN) || value > static_cast<double>(INT64_MAX))
        {
            throw std::runtime_error(error_prefix + ": integer is out of int64 range.");
        }
        return static_cast<int64_t>(value);
    }
};

class Parser {
public:
    Parser(std::string text, std::string error_prefix)
        : text_(std::move(text)),
          error_prefix_(std::move(error_prefix))
    {
    }

    Value parse()
    {
        skip_whitespace();
        Value value = parse_value();
        skip_whitespace();
        if (!eof())
        {
            fail("unexpected trailing content");
        }
        return value;
    }

private:
    static bool is_hex_digit(char ch)
    {
        return (ch >= '0' && ch <= '9')
            || (ch >= 'a' && ch <= 'f')
            || (ch >= 'A' && ch <= 'F');
    }

    static int hex_value(char ch)
    {
        if (ch >= '0' && ch <= '9')
        {
            return ch - '0';
        }
        if (ch >= 'a' && ch <= 'f')
        {
            return 10 + (ch - 'a');
        }
        return 10 + (ch - 'A');
    }

    bool eof() const
    {
        return pos_ >= text_.size();
    }

    char peek() const
    {
        if (eof())
        {
            fail("unexpected end of input");
        }
        return text_[pos_];
    }

    char get()
    {
        const char ch = peek();
        ++pos_;
        return ch;
    }

    void skip_whitespace()
    {
        while (!eof() && std::isspace(static_cast<unsigned char>(text_[pos_])) != 0)
        {
            ++pos_;
        }
    }

    bool consume_if(char ch)
    {
        if (!eof() && text_[pos_] == ch)
        {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect_char(char ch)
    {
        if (!consume_if(ch))
        {
            fail(std::string("expected '") + ch + "'");
        }
    }

    void expect_literal(const char* literal)
    {
        const size_t start = pos_;
        size_t i = 0;
        while (literal[i] != '\0')
        {
            if (eof() || text_[pos_] != literal[i])
            {
                pos_ = start;
                fail(std::string("expected literal ") + literal);
            }
            ++pos_;
            ++i;
        }
    }

    std::string parse_string()
    {
        expect_char('"');
        std::string out;
        while (true)
        {
            if (eof())
            {
                fail("unterminated string");
            }

            const char ch = get();
            if (ch == '"')
            {
                break;
            }

            if (ch == '\\')
            {
                if (eof())
                {
                    fail("unterminated escape sequence");
                }
                const char esc = get();
                switch (esc)
                {
                    case '"':
                        out.push_back('"');
                        break;
                    case '\\':
                        out.push_back('\\');
                        break;
                    case '/':
                        out.push_back('/');
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    case 'u':
                    {
                        if (pos_ + 4 > text_.size())
                        {
                            fail("invalid unicode escape");
                        }

                        int code_point = 0;
                        for (int i = 0; i < 4; ++i)
                        {
                            const char hex = text_[pos_ + static_cast<size_t>(i)];
                            if (!is_hex_digit(hex))
                            {
                                fail("invalid unicode escape");
                            }
                            code_point = (code_point << 4) | hex_value(hex);
                        }
                        pos_ += 4;

                        if (code_point <= 0x7F)
                        {
                            out.push_back(static_cast<char>(code_point));
                        }
                        else if (code_point <= 0x7FF)
                        {
                            out.push_back(static_cast<char>(0xC0 | ((code_point >> 6) & 0x1F)));
                            out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                        }
                        else
                        {
                            out.push_back(static_cast<char>(0xE0 | ((code_point >> 12) & 0x0F)));
                            out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
                            out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
                        }
                        break;
                    }
                    default:
                        fail("unsupported escape character");
                }
                continue;
            }

            out.push_back(ch);
        }

        return out;
    }

    Value parse_number()
    {
        const char* begin = text_.c_str() + static_cast<std::ptrdiff_t>(pos_);
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin)
        {
            fail("invalid number");
        }

        const size_t consumed = static_cast<size_t>(end - begin);
        pos_ += consumed;
        return Value::make_number(value);
    }

    Value parse_array()
    {
        expect_char('[');
        skip_whitespace();

        std::vector<Value> values;
        if (consume_if(']'))
        {
            return Value::make_array(std::move(values));
        }

        while (true)
        {
            skip_whitespace();
            values.push_back(parse_value());
            skip_whitespace();

            if (consume_if(']'))
            {
                break;
            }
            expect_char(',');
        }

        return Value::make_array(std::move(values));
    }

    Value parse_object()
    {
        expect_char('{');
        skip_whitespace();

        std::unordered_map<std::string, std::unique_ptr<Value>> object;
        if (consume_if('}'))
        {
            return Value::make_object(std::move(object));
        }

        while (true)
        {
            skip_whitespace();
            if (peek() != '"')
            {
                fail("object key must be string");
            }

            const std::string key = parse_string();
            skip_whitespace();
            expect_char(':');
            skip_whitespace();
            object[key] = std::make_unique<Value>(parse_value());
            skip_whitespace();

            if (consume_if('}'))
            {
                break;
            }
            expect_char(',');
        }

        return Value::make_object(std::move(object));
    }

    Value parse_value()
    {
        if (eof())
        {
            fail("unexpected end while parsing value");
        }

        const char ch = peek();
        switch (ch)
        {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return Value::make_string(parse_string());
            case 't':
                expect_literal("true");
                return Value::make_bool(true);
            case 'f':
                expect_literal("false");
                return Value::make_bool(false);
            case 'n':
                expect_literal("null");
                return Value::make_null();
            default:
                if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch)) != 0)
                {
                    return parse_number();
                }
                fail("unexpected character while parsing value");
        }
    }

    [[noreturn]] void fail(const std::string& message) const
    {
        throw std::runtime_error(
            error_prefix_ + ": " + message + " at byte offset " + std::to_string(pos_) + ".");
    }

    std::string text_;
    std::string error_prefix_;
    size_t pos_ = 0;
};

inline Value parse(const std::string& text, const std::string& error_prefix)
{
    Parser parser(text, error_prefix);
    return parser.parse();
}

inline const Value* find_object_field(const Value& object,
                                      const std::string& key,
                                      const std::string& error_prefix)
{
    const auto& map = object.as_object(error_prefix);
    const auto it = map.find(key);
    if (it == map.end())
    {
        return nullptr;
    }
    return it->second.get();
}

inline const Value& require_object_field(const Value& object,
                                         const std::string& key,
                                         const std::string& error_prefix)
{
    const auto* value = find_object_field(object, key, error_prefix);
    if (value == nullptr)
    {
        throw std::runtime_error(error_prefix + ": missing required key: " + key);
    }
    return *value;
}

} // namespace hf_json
} // namespace tiny_llm
