package adaboost

type AdaBoostClassifier struct {
	nEstimators int
}

func NewAdaBoostClassifier(nEstimators int) *AdaBoostClassifier {
	return &AdaBoostClassifier{
		nEstimators: nEstimators,
	}
}
