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

#pragma once

#ifdef TRACCC_USE_ACTS_LOGGER
#include <Acts/Utilities/Logger.hpp>

namespace traccc {
namespace logging = ::Acts::Logging;

using Logger = ::Acts::Logger;

std::unique_ptr<const Logger> getDefaultLogger(
    const std::string& name, const logging::Level& lvl,
    std::ostream* log_stream = &std::cout);

const Logger& getDummyLogger();
}  // namespace traccc

#define TRACCC_LOCAL_LOGGER(x) ACTS_LOCAL_LOGGER(x)
#define TRACCC_LOG(level, x) ACTS_LOG(level, log)
#define TRACCC_VERBOSE(x) ACTS_VERBOSE(x)
#define TRACCC_DEBUG(x) ACTS_DEBUG(x)
#define TRACCC_INFO(x) ACTS_INFO(x)
#define TRACCC_WARNING(x) ACTS_WARNING(x)
#define TRACCC_ERROR(x) ACTS_ERROR(x)
#define TRACCC_FATAL(x) ACTS_FATAL(x)
#else
// STL include(s)
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#define TRACCC_LOCAL_LOGGER(log_object)                                     \
    struct __local_acts_logger {                                            \
        __local_acts_logger(std::unique_ptr<const ::traccc::Logger> logger) \
            : m_logger(std::move(logger)) {}                                \
                                                                            \
        const ::traccc::Logger& operator()() const { return *m_logger; }    \
                                                                            \
        std::unique_ptr<const ::traccc::Logger> m_logger;                   \
    };                                                                      \
    __local_acts_logger logger(log_object);                                 \
    do {                                                                    \
        (void)logger;                                                       \
    } while (0)

#define TRACCC_LOG(level, x)               \
    do {                                   \
        if (logger().doPrint(level)) {     \
            std::ostringstream os;         \
            os << x;                       \
            logger().log(level, os.str()); \
        }                                  \
    } while (0)

#define TRACCC_VERBOSE(x) TRACCC_LOG(::traccc::logging::VERBOSE, x)

#define TRACCC_DEBUG(x) TRACCC_LOG(::traccc::logging::DEBUG, x)

#define TRACCC_INFO(x) TRACCC_LOG(::traccc::logging::INFO, x)

#define TRACCC_WARNING(x) TRACCC_LOG(::traccc::logging::WARNING, x)

#define TRACCC_ERROR(x) TRACCC_LOG(::traccc::logging::ERROR, x)

#define TRACCC_FATAL(x) TRACCC_LOG(::traccc::logging::FATAL, x)

namespace traccc {
namespace logging {
enum Level { VERBOSE = 0, DEBUG, INFO, WARNING, ERROR, FATAL, MAX };

inline std::string_view levelName(Level level);

class OutputPrintPolicy {
    public:
    virtual ~OutputPrintPolicy();
    virtual void flush(const Level& lvl, const std::string& input) = 0;
    virtual const std::string& name() const = 0;
    virtual std::unique_ptr<OutputPrintPolicy> clone(
        const std::string& name) const = 0;
};

class OutputFilterPolicy {
    public:
    virtual ~OutputFilterPolicy();
    virtual bool doPrint(const Level& lvl) const = 0;
    virtual Level level() const = 0;
    virtual std::unique_ptr<OutputFilterPolicy> clone(Level level) const = 0;
};

class DefaultFilterPolicy final : public OutputFilterPolicy {
    public:
    explicit DefaultFilterPolicy(Level lvl);
    ~DefaultFilterPolicy() override;
    bool doPrint(const Level& lvl) const override;
    Level level() const override;
    std::unique_ptr<OutputFilterPolicy> clone(Level level) const override;

    private:
    Level m_level;
};

class OutputDecorator : public OutputPrintPolicy {
    public:
    explicit OutputDecorator(std::unique_ptr<OutputPrintPolicy> wrappee);

    void flush(const Level& lvl, const std::string& input) override;

    const std::string& name() const override;

    protected:
    std::unique_ptr<OutputPrintPolicy> m_wrappee;
};

class NamedOutputDecorator final : public OutputDecorator {
    public:
    NamedOutputDecorator(std::unique_ptr<OutputPrintPolicy> wrappee,
                         const std::string& name, int maxWidth = 15);
    void flush(const Level& lvl, const std::string& input) override;
    std::unique_ptr<OutputPrintPolicy> clone(
        const std::string& name) const override;
    const std::string& name() const override;

    private:
    std::string m_name;
    int m_maxWidth;
};

class TimedOutputDecorator final : public OutputDecorator {
    public:
    TimedOutputDecorator(std::unique_ptr<OutputPrintPolicy> wrappee,
                         const std::string& format = "%X");
    void flush(const Level& lvl, const std::string& input) override;
    std::unique_ptr<OutputPrintPolicy> clone(
        const std::string& name) const override;

    private:
    std::string now() const {
        char buffer[20];
        time_t t{};
        std::time(&t);
        struct tm tbuf {};
        std::strftime(buffer, sizeof(buffer), m_format.c_str(),
                      localtime_r(&t, &tbuf));
        return buffer;
    }

    std::string m_format;
};

class LevelOutputDecorator final : public OutputDecorator {
    public:
    explicit LevelOutputDecorator(std::unique_ptr<OutputPrintPolicy> wrappee);
    void flush(const Level& lvl, const std::string& input) override;

    std::unique_ptr<OutputPrintPolicy> clone(
        const std::string& name) const override;

    private:
    std::string toString(const Level& lvl) const;
};

class DefaultPrintPolicy final : public OutputPrintPolicy {
    public:
    explicit DefaultPrintPolicy(std::ostream* out = &std::cout);

    void flush(const Level&, const std::string& input) final;
    const std::string& name() const override;

    std::unique_ptr<OutputPrintPolicy> clone(
        const std::string& /*name*/) const override;

    private:
    std::ostream* m_out;
};
}  // namespace logging

class Logger {
    public:
    Logger(std::unique_ptr<logging::OutputPrintPolicy> pPrint,
           std::unique_ptr<logging::OutputFilterPolicy> pFilter);

    bool doPrint(const logging::Level& lvl) const;

    void log(const logging::Level& lvl, const std::string& input) const;

    const logging::OutputPrintPolicy& printPolicy() const;

    const logging::OutputFilterPolicy& filterPolicy() const;

    logging::Level level() const;

    const std::string& name() const;

    std::unique_ptr<Logger> clone(
        const std::optional<std::string>& _name = std::nullopt,
        const std::optional<logging::Level>& _level = std::nullopt) const;

    std::unique_ptr<Logger> clone(logging::Level _level) const;

    std::unique_ptr<Logger> cloneWithSuffix(
        const std::string& suffix,
        std::optional<logging::Level> _level = std::nullopt) const;

    const Logger& operator()() const;

    private:
    std::unique_ptr<logging::OutputPrintPolicy> m_printPolicy;
    std::unique_ptr<logging::OutputFilterPolicy> m_filterPolicy;
};

std::unique_ptr<const Logger> getDefaultLogger(
    const std::string& name, const logging::Level& lvl,
    std::ostream* log_stream = &std::cout);

const Logger& getDummyLogger();
}  // namespace traccc
#endif
