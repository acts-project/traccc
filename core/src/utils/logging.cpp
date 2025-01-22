/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2021-2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

/*
 * This file was copied from the ACTS repository, which is also available
 * under the Mozilla Public License Version 2.0
 */

#include "traccc/utils/logging.hpp"

#ifndef TRACCC_USE_ACTS_LOGGER

#include <algorithm>
#include <cstdlib>

namespace traccc {

namespace logging {

namespace {
class NeverFilterPolicy final : public OutputFilterPolicy {
    public:
    ~NeverFilterPolicy() override = default;

    bool doPrint(const Level&) const override { return false; }

    Level level() const override { return Level::MAX; }

    std::unique_ptr<OutputFilterPolicy> clone(Level /*level*/) const override {
        return std::make_unique<NeverFilterPolicy>();
    }
};

class DummyPrintPolicy final : public OutputPrintPolicy {
    public:
    void flush(const Level& /*lvl*/, const std::string& /*input*/) override {}

    const std::string& name() const override {
        const static std::string s_name = "Dummy";
        return s_name;
    }

    std::unique_ptr<OutputPrintPolicy> clone(
        const std::string& /*name*/) const override {
        return std::make_unique<DummyPrintPolicy>();
    }
};

std::unique_ptr<const Logger> makeDummyLogger() {
    using namespace logging;
    auto output = std::make_unique<DummyPrintPolicy>();
    auto print = std::make_unique<NeverFilterPolicy>();
    return std::make_unique<const Logger>(std::move(output), std::move(print));
}

}  // namespace

inline std::string_view levelName(Level level) {
    switch (level) {
        case Level::VERBOSE:
            return "VERBOSE";
        case Level::DEBUG:
            return "DEBUG";
        case Level::INFO:
            return "INFO";
        case Level::WARNING:
            return "WARNING";
        case Level::ERROR:
            return "ERROR";
        case Level::FATAL:
            return "FATAL";
        case Level::MAX:
            return "MAX";
        default:
            throw std::invalid_argument{"Unknown level"};
    }
}

OutputPrintPolicy::~OutputPrintPolicy() = default;
OutputFilterPolicy::~OutputFilterPolicy() = default;

DefaultFilterPolicy::DefaultFilterPolicy(Level lvl) : m_level(lvl) {}
DefaultFilterPolicy::~DefaultFilterPolicy() = default;
bool DefaultFilterPolicy::doPrint(const Level& lvl) const {
    return m_level <= lvl;
}
Level DefaultFilterPolicy::level() const {
    return m_level;
}
std::unique_ptr<OutputFilterPolicy> DefaultFilterPolicy::clone(
    Level level) const {
    return std::make_unique<DefaultFilterPolicy>(level);
}

OutputDecorator::OutputDecorator(std::unique_ptr<OutputPrintPolicy> wrappee)
    : m_wrappee(std::move(wrappee)) {}
void OutputDecorator::flush(const Level& lvl, const std::string& input) {
    m_wrappee->flush(lvl, input);
}
const std::string& OutputDecorator::name() const {
    return m_wrappee->name();
}

NamedOutputDecorator::NamedOutputDecorator(
    std::unique_ptr<OutputPrintPolicy> wrappee, const std::string& name,
    int maxWidth)
    : OutputDecorator(std::move(wrappee)), m_name(name), m_maxWidth(maxWidth) {}
void NamedOutputDecorator::flush(const Level& lvl, const std::string& input) {
    std::ostringstream os;
    os << std::left << std::setw(m_maxWidth)
       << m_name.substr(0,
                        static_cast<std::size_t>(std::max(0, m_maxWidth - 3)))
       << input;
    OutputDecorator::flush(lvl, os.str());
}
std::unique_ptr<OutputPrintPolicy> NamedOutputDecorator::clone(
    const std::string& name) const {
    return std::make_unique<NamedOutputDecorator>(m_wrappee->clone(name), name,
                                                  m_maxWidth);
}
const std::string& NamedOutputDecorator::name() const {
    return m_name;
}

TimedOutputDecorator::TimedOutputDecorator(
    std::unique_ptr<OutputPrintPolicy> wrappee, const std::string& format)
    : OutputDecorator(std::move(wrappee)), m_format(format) {}
void TimedOutputDecorator::flush(const Level& lvl, const std::string& input) {
    std::ostringstream os;
    os << std::left << std::setw(12) << now() << input;
    OutputDecorator::flush(lvl, os.str());
}
std::unique_ptr<OutputPrintPolicy> TimedOutputDecorator::clone(
    const std::string& name) const {
    return std::make_unique<TimedOutputDecorator>(m_wrappee->clone(name),
                                                  m_format);
}

LevelOutputDecorator::LevelOutputDecorator(
    std::unique_ptr<OutputPrintPolicy> wrappee)
    : OutputDecorator(std::move(wrappee)) {}
void LevelOutputDecorator::flush(const Level& lvl, const std::string& input) {
    std::ostringstream os;
    os << std::left << std::setw(10) << toString(lvl) << input;
    OutputDecorator::flush(lvl, os.str());
}
std::unique_ptr<OutputPrintPolicy> LevelOutputDecorator::clone(
    const std::string& name) const {
    return std::make_unique<LevelOutputDecorator>(m_wrappee->clone(name));
}
std::string LevelOutputDecorator::toString(const Level& lvl) const {
    static const char* const buffer[] = {"VERBOSE", "DEBUG", "INFO",
                                         "WARNING", "ERROR", "FATAL"};
    return buffer[lvl];
}

DefaultPrintPolicy::DefaultPrintPolicy(std::ostream* out) : m_out(out) {}
void DefaultPrintPolicy::flush(const Level&, const std::string& input) {
    (*m_out) << input << std::endl;
}
const std::string& DefaultPrintPolicy::name() const {
    throw std::runtime_error{
        "Default print policy doesn't have a name. Is there no named "
        "output in "
        "the decorator chain?"};
}
std::unique_ptr<OutputPrintPolicy> DefaultPrintPolicy::clone(
    const std::string& /*name*/) const {
    return std::make_unique<DefaultPrintPolicy>(m_out);
}

}  // namespace logging

Logger::Logger(std::unique_ptr<logging::OutputPrintPolicy> pPrint,
               std::unique_ptr<logging::OutputFilterPolicy> pFilter)
    : m_printPolicy(std::move(pPrint)), m_filterPolicy(std::move(pFilter)) {}

bool Logger::doPrint(const logging::Level& lvl) const {
    return m_filterPolicy->doPrint(lvl);
}

void Logger::log(const logging::Level& lvl, const std::string& input) const {
    if (doPrint(lvl)) {
        m_printPolicy->flush(lvl, input);
    }
}

const logging::OutputPrintPolicy& Logger::printPolicy() const {
    return *m_printPolicy;
}

const logging::OutputFilterPolicy& Logger::filterPolicy() const {
    return *m_filterPolicy;
}

logging::Level Logger::level() const {
    return m_filterPolicy->level();
}

const std::string& Logger::name() const {
    return m_printPolicy->name();
}

std::unique_ptr<Logger> Logger::clone(
    const std::optional<std::string>& _name,
    const std::optional<logging::Level>& _level) const {
    return std::make_unique<Logger>(
        m_printPolicy->clone(_name.value_or(name())),
        m_filterPolicy->clone(_level.value_or(level())));
}

std::unique_ptr<Logger> Logger::clone(logging::Level _level) const {
    return clone(std::nullopt, _level);
}

std::unique_ptr<Logger> Logger::cloneWithSuffix(
    const std::string& suffix, std::optional<logging::Level> _level) const {
    return clone(name() + suffix, _level.value_or(level()));
}

const Logger& Logger::operator()() const {
    return *this;
}

std::unique_ptr<const Logger> getDefaultLogger(const std::string& name,
                                               const logging::Level& lvl,
                                               std::ostream* log_stream) {
    using namespace logging;
    auto output = std::make_unique<LevelOutputDecorator>(
        std::make_unique<NamedOutputDecorator>(
            std::make_unique<TimedOutputDecorator>(
                std::make_unique<DefaultPrintPolicy>(log_stream)),
            name));
    auto print = std::make_unique<DefaultFilterPolicy>(lvl);
    return std::make_unique<const Logger>(std::move(output), std::move(print));
}

const Logger& getDummyLogger() {
    static const std::unique_ptr<const Logger> logger =
        logging::makeDummyLogger();

    return *logger;
}
}  // namespace traccc
#else
namespace traccc {
std::unique_ptr<const Logger> getDefaultLogger(const std::string& name,
                                               const logging::Level& lvl,
                                               std::ostream* log_stream) {
    return ::Acts::getDefaultLogger(name, lvl, log_stream);
}

const Logger& getDummyLogger() {
    return ::Acts::getDummyLogger();
}
}  // namespace traccc
#endif
