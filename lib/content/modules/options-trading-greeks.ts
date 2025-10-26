import { Module } from '../../types';

// Import sections
import { optionsFundamentals } from '@/lib/content/sections/options-trading-greeks/options-fundamentals';
import { callPutDeepDive } from '@/lib/content/sections/options-trading-greeks/call-put-deep-dive';
import { optionsPricingBlackScholes } from '@/lib/content/sections/options-trading-greeks/options-pricing-black-scholes';
import { greeksDeltaGammaThetaVegaRho } from '@/lib/content/sections/options-trading-greeks/greeks-delta-gamma-theta-vega-rho';
import { impliedVolatility } from '@/lib/content/sections/options-trading-greeks/implied-volatility';
import { optionsTradingStrategies } from '@/lib/content/sections/options-trading-greeks/options-trading-strategies';
import { coveredCallsProtectivePuts } from '@/lib/content/sections/options-trading-greeks/covered-calls-protective-puts';
import { spreadsStrategies } from '@/lib/content/sections/options-trading-greeks/spreads-strategies';
import { straddlesStrangles } from '@/lib/content/sections/options-trading-greeks/straddles-strangles';
import { optionsMarketMaking } from '@/lib/content/sections/options-trading-greeks/options-market-making';
import { volatilityTrading } from '@/lib/content/sections/options-trading-greeks/volatility-trading';
import { portfolioGreeksManagement } from '@/lib/content/sections/options-trading-greeks/portfolio-greeks-management';
import { moduleProjectOptionsPlatform } from '@/lib/content/sections/options-trading-greeks/module-project-options-platform';

// Import quizzes
import { optionsFundamentalsQuiz } from '@/lib/content/quizzes/options-trading-greeks/options-fundamentals';
import { callPutDeepDiveQuiz } from '@/lib/content/quizzes/options-trading-greeks/call-put-deep-dive';
import { optionsPricingBlackScholesQuiz } from '@/lib/content/quizzes/options-trading-greeks/options-pricing-black-scholes';
import { greeksDeltaGammaThetaVegaRhoQuiz } from '@/lib/content/quizzes/options-trading-greeks/greeks-delta-gamma-theta-vega-rho';
import { impliedVolatilityQuiz } from '@/lib/content/quizzes/options-trading-greeks/implied-volatility';
import { optionsTradingStrategiesQuiz } from '@/lib/content/quizzes/options-trading-greeks/options-trading-strategies';
import { coveredCallsProtectivePutsQuiz } from '@/lib/content/quizzes/options-trading-greeks/covered-calls-protective-puts';
import { spreadsStrategiesQuiz } from '@/lib/content/quizzes/options-trading-greeks/spreads-strategies';
import { straddlesStranglesQuiz } from '@/lib/content/quizzes/options-trading-greeks/straddles-strangles';
import { optionsMarketMakingQuiz } from '@/lib/content/quizzes/options-trading-greeks/options-market-making';
import { volatilityTradingQuiz } from '@/lib/content/quizzes/options-trading-greeks/volatility-trading';
import { portfolioGreeksManagementQuiz } from '@/lib/content/quizzes/options-trading-greeks/portfolio-greeks-management';
import { moduleProjectOptionsPlatformQuiz } from '@/lib/content/quizzes/options-trading-greeks/module-project-options-platform';

// Import multiple choice
import { optionsFundamentalsMC } from '@/lib/content/multiple-choice/options-trading-greeks/options-fundamentals';
import { callPutDeepDiveMC } from '@/lib/content/multiple-choice/options-trading-greeks/call-put-deep-dive';
import { optionsPricingBlackScholesMC } from '@/lib/content/multiple-choice/options-trading-greeks/options-pricing-black-scholes';
import { greeksDeltaGammaThetaVegaRhoMC } from '@/lib/content/multiple-choice/options-trading-greeks/greeks-delta-gamma-theta-vega-rho';
import { impliedVolatilityMC } from '@/lib/content/multiple-choice/options-trading-greeks/implied-volatility';
import { optionsTradingStrategiesMC } from '@/lib/content/multiple-choice/options-trading-greeks/options-trading-strategies';
import { coveredCallsProtectivePutsMC } from '@/lib/content/multiple-choice/options-trading-greeks/covered-calls-protective-puts';
import { spreadsStrategiesMC } from '@/lib/content/multiple-choice/options-trading-greeks/spreads-strategies';
import { straddlesStranglesMC } from '@/lib/content/multiple-choice/options-trading-greeks/straddles-strangles';
import { optionsMarketMakingMC } from '@/lib/content/multiple-choice/options-trading-greeks/options-market-making';
import { volatilityTradingMC } from '@/lib/content/multiple-choice/options-trading-greeks/volatility-trading';
import { portfolioGreeksManagementMC } from '@/lib/content/multiple-choice/options-trading-greeks/portfolio-greeks-management';
import { moduleProjectOptionsPlatformMC } from '@/lib/content/multiple-choice/options-trading-greeks/module-project-options-platform';

// Transform functions
const transformQuiz = (quiz: any) => ({
    id: quiz.id,
    question: quiz.question,
    sampleAnswer: quiz.sampleAnswer,
    keyPoints: quiz.keyPoints,
});

const transformMC = (mc: any) => ({
    id: mc.id,
    question: mc.question,
    options: mc.options,
    correctAnswer: mc.correctAnswer,
    explanation: mc.explanation,
});

export const optionsTradingGreeksModule: Module = {
    id: 'options-trading-greeks',
    title: 'Options Trading & Greeks',
    description:
        'Master options pricing, Greeks, volatility trading, and portfolio management for professional options trading',
    sections: [
        {
            id: optionsFundamentals.id,
            title: optionsFundamentals.title,
            content: optionsFundamentals.content,
            quiz: optionsFundamentalsQuiz.map(transformQuiz),
            multipleChoice: optionsFundamentalsMC.map(transformMC),
        },
        {
            id: callPutDeepDive.id,
            title: callPutDeepDive.title,
            content: callPutDeepDive.content,
            quiz: callPutDeepDiveQuiz.map(transformQuiz),
            multipleChoice: callPutDeepDiveMC.map(transformMC),
        },
        {
            id: optionsPricingBlackScholes.id,
            title: optionsPricingBlackScholes.title,
            content: optionsPricingBlackScholes.content,
            quiz: optionsPricingBlackScholesQuiz.map(transformQuiz),
            multipleChoice: optionsPricingBlackScholesMC.map(transformMC),
        },
        {
            id: greeksDeltaGammaThetaVegaRho.id,
            title: greeksDeltaGammaThetaVegaRho.title,
            content: greeksDeltaGammaThetaVegaRho.content,
            quiz: greeksDeltaGammaThetaVegaRhoQuiz.map(transformQuiz),
            multipleChoice: greeksDeltaGammaThetaVegaRhoMC.map(transformMC),
        },
        {
            id: impliedVolatility.id,
            title: impliedVolatility.title,
            content: impliedVolatility.content,
            quiz: impliedVolatilityQuiz.map(transformQuiz),
            multipleChoice: impliedVolatilityMC.map(transformMC),
        },
        {
            id: optionsTradingStrategies.id,
            title: optionsTradingStrategies.title,
            content: optionsTradingStrategies.content,
            quiz: optionsTradingStrategiesQuiz.map(transformQuiz),
            multipleChoice: optionsTradingStrategiesMC.map(transformMC),
        },
        {
            id: coveredCallsProtectivePuts.id,
            title: coveredCallsProtectivePuts.title,
            content: coveredCallsProtectivePuts.content,
            quiz: coveredCallsProtectivePutsQuiz.map(transformQuiz),
            multipleChoice: coveredCallsProtectivePutsMC.map(transformMC),
        },
        {
            id: spreadsStrategies.id,
            title: spreadsStrategies.title,
            content: spreadsStrategies.content,
            quiz: spreadsStrategiesQuiz.map(transformQuiz),
            multipleChoice: spreadsStrategiesMC.map(transformMC),
        },
        {
            id: straddlesStrangles.id,
            title: straddlesStrangles.title,
            content: straddlesStrangles.content,
            quiz: straddlesStranglesQuiz.map(transformQuiz),
            multipleChoice: straddlesStranglesMC.map(transformMC),
        },
        {
            id: optionsMarketMaking.id,
            title: optionsMarketMaking.title,
            content: optionsMarketMaking.content,
            quiz: optionsMarketMakingQuiz.map(transformQuiz),
            multipleChoice: optionsMarketMakingMC.map(transformMC),
        },
        {
            id: volatilityTrading.id,
            title: volatilityTrading.title,
            content: volatilityTrading.content,
            quiz: volatilityTradingQuiz.map(transformQuiz),
            multipleChoice: volatilityTradingMC.map(transformMC),
        },
        {
            id: portfolioGreeksManagement.id,
            title: portfolioGreeksManagement.title,
            content: portfolioGreeksManagement.content,
            quiz: portfolioGreeksManagementQuiz.map(transformQuiz),
            multipleChoice: portfolioGreeksManagementMC.map(transformMC),
        },
        {
            id: moduleProjectOptionsPlatform.id,
            title: moduleProjectOptionsPlatform.title,
            content: moduleProjectOptionsPlatform.content,
            quiz: moduleProjectOptionsPlatformQuiz.map(transformQuiz),
            multipleChoice: moduleProjectOptionsPlatformMC.map(transformMC),
        },
    ],
    quizzes: [
        ...optionsFundamentalsQuiz.map(transformQuiz),
        ...callPutDeepDiveQuiz.map(transformQuiz),
        ...optionsPricingBlackScholesQuiz.map(transformQuiz),
        ...greeksDeltaGammaThetaVegaRhoQuiz.map(transformQuiz),
        ...impliedVolatilityQuiz.map(transformQuiz),
        ...optionsTradingStrategiesQuiz.map(transformQuiz),
        ...coveredCallsProtectivePutsQuiz.map(transformQuiz),
        ...spreadsStrategiesQuiz.map(transformQuiz),
        ...straddlesStranglesQuiz.map(transformQuiz),
        ...optionsMarketMakingQuiz.map(transformQuiz),
        ...volatilityTradingQuiz.map(transformQuiz),
        ...portfolioGreeksManagementQuiz.map(transformQuiz),
        ...moduleProjectOptionsPlatformQuiz.map(transformQuiz),
    ],
    multipleChoiceQuestions: [
        ...optionsFundamentalsMC.map(transformMC),
        ...callPutDeepDiveMC.map(transformMC),
        ...optionsPricingBlackScholesMC.map(transformMC),
        ...greeksDeltaGammaThetaVegaRhoMC.map(transformMC),
        ...impliedVolatilityMC.map(transformMC),
        ...optionsTradingStrategiesMC.map(transformMC),
        ...coveredCallsProtectivePutsMC.map(transformMC),
        ...spreadsStrategiesMC.map(transformMC),
        ...straddlesStranglesMC.map(transformMC),
        ...optionsMarketMakingMC.map(transformMC),
        ...volatilityTradingMC.map(transformMC),
        ...portfolioGreeksManagementMC.map(transformMC),
        ...moduleProjectOptionsPlatformMC.map(transformMC),
    ],
};

