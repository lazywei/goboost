package adaboost

import (
	"math"

	"github.com/gonum/matrix/mat64"
)

type AdaBoostClassifier struct {
	nEstimators int
	nSamples    int
	nFeatures   int
	clfs        []*baseLearner
}

type baseLearner struct {
	classifier *DecisionStump
	weight     float64
}

func NewAdaBoostClassifier(nEstimators int) *AdaBoostClassifier {
	return &AdaBoostClassifier{
		nEstimators: nEstimators,
	}
}

func (abc *AdaBoostClassifier) Fit(X *mat64.Dense, y []int) error {
	nRows, nCols := X.Dims()
	abc.nSamples = nRows
	abc.nFeatures = nCols

	sampleWeight := make([]float64, nRows)
	for i := range sampleWeight {
		sampleWeight[i] = 1
	}

	for i := 0; i < abc.nEstimators; i++ {
		ds := NewDecisionStump(10)
		ds.Fit(X, y, sampleWeight)

		errRate := ErrRate(y, ds.Predict(X), sampleWeight)
		alpha := math.Log((1-errRate)/errRate) / 2
		abc.clfs = append(abc.clfs, &baseLearner{classifier: ds, weight: alpha})

		sampleWeight = abc.updateWeight(X, y, alpha, sampleWeight)
	}

	return nil
}

func (abc *AdaBoostClassifier) RawPredict(X *mat64.Dense) []float64 {
	yRawPred := make([]float64, abc.nSamples)
	for _, clf := range abc.clfs {

		for i, x := range clf.classifier.Predict(X) {
			yRawPred[i] = yRawPred[i] + float64(x)*clf.weight
		}

	}

	return yRawPred
}

func (abc *AdaBoostClassifier) Predict(X *mat64.Dense) []int {
	yPred := make([]int, abc.nSamples)
	for i, val := range abc.RawPredict(X) {
		if val > 0 {
			yPred[i] = 1
		} else {
			yPred[i] = -1
		}
	}

	return yPred
}

func (abc *AdaBoostClassifier) updateWeight(X *mat64.Dense, y []int, alpha float64, currentWeight []float64) []float64 {
	newWeight := currentWeight
	for i, rawY := range abc.RawPredict(X) {
		newWeight[i] = currentWeight[i] * math.Exp(-1*float64(y[i])*rawY)
	}

	return newWeight
}
