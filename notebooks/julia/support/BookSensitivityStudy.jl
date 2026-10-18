module BookSensitivityStudy

using CSV, DataFrames, LinearAlgebra, Statistics
using ..BookSupport: book_root
using ..BookCausalStudy: frame

function sensitivity_ols(outcome,predictors...)
    design=hcat(ones(length(outcome)),predictors...)
    decomposition=qr(design)
    coefficients=decomposition\outcome
    residual=outcome-design*coefficients
    ssr=sum(abs2,residual)
    degrees=length(outcome)-size(design,2)
    inverse=inv(Matrix(decomposition.R))
    covariance=ssr/degrees*(inverse*inverse')
    errors=sqrt.(diag(covariance))
    return (;coefficients,errors,ssr,degrees,r2=1-ssr/sum(abs2,outcome.-mean(outcome)),statistics=coefficients./errors)
end

sensitivity_partial_r2(restricted,expanded)=1-expanded.ssr/restricted.ssr

function sensitivity_estimate(exposure_strength,outcome_strength,estimate,se,degrees;reduce=true)
    bias=se*sqrt(degrees)*sqrt(outcome_strength*exposure_strength/(1-exposure_strength))
    return sign(estimate)*(abs(estimate)+(reduce ? -bias : bias))
end

sensitivity_error(exposure_strength,outcome_strength,se,degrees)=sqrt((1-outcome_strength)/(1-exposure_strength))*se*sqrt(degrees/(degrees-1))

function zero_robustness(statistic,degrees)
    magnitude=abs(statistic)/sqrt(degrees)
    return .5*(sqrt(magnitude^4+4magnitude^2)-magnitude^2)
end

function sensitivity_counties()
    raw=CSV.read(joinpath(book_root(),"data","public","callaway_santanna","mpdta.csv"),DataFrame)
    records=[]
    for identifier in sort(unique(raw.countyreal))
        observations=raw[raw.countyreal.==identifier,:]
        values=Dict(Int(row.year)=>row for row in eachrow(observations))
        first_treat=first(observations.first_treat)
        push!(records,(identifier,Int(first_treat>0),values[2003].lemp,values[2006].lemp,values[2007].lemp,values[2003].lpop,values[2007].lemp-values[2003].lemp,Int(values[2007].lemp>values[2006].lemp)))
    end
    return raw,frame(["County identifier","Treatment group","Log employment in 2003","Log employment in 2006","Log employment in 2007","Log population in 2003","Employment change, 2003 to 2007","Employment increased, 2006 to 2007"],records)
end

export sensitivity_ols, sensitivity_partial_r2, sensitivity_estimate,
    sensitivity_error, zero_robustness, sensitivity_counties

end
