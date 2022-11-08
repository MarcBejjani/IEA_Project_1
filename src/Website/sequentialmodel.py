import joblib

SVM_stage1 = joblib.load('FinalEnsemble1/SVM_stage1.npy')

SVM1_stage2 = joblib.load('FinalEnsemble1/SVM1_stage2.npy')
SVM2_stage2 = joblib.load('FinalEnsemble1/SVM2_stage2.npy')

SVM1_stage3 = joblib.load('FinalEnsemble1/SVM1_stage3.npy')
SVM2_stage3 = joblib.load('FinalEnsemble1/SVM2_stage3.npy')
SVM3_stage3 = joblib.load('FinalEnsemble1/SVM3_stage3.npy')
SVM4_stage3 = joblib.load('FinalEnsemble1/SVM4_stage3.npy')

SVM1_stage4 = joblib.load('FinalEnsemble1/SVM1_stage4.npy')
SVM2_stage4 = joblib.load('FinalEnsemble1/SVM2_stage4.npy')
SVM3_stage4 = joblib.load('FinalEnsemble1/SVM3_stage4.npy')
SVM4_stage4 = joblib.load('FinalEnsemble1/SVM4_stage4.npy')
SVM5_stage4 = joblib.load('FinalEnsemble1/SVM5_stage4.npy')
SVM6_stage4 = joblib.load('FinalEnsemble1/SVM6_stage4.npy')
SVM7_stage4 = joblib.load('FinalEnsemble1/SVM7_stage4.npy')
SVM8_stage4 = joblib.load('FinalEnsemble1/SVM8_stage4.npy')

FinalStageSVM1 = joblib.load('FinalEnsemble1/FinalStageSVM1.npy')
FinalStageSVM2 = joblib.load('FinalEnsemble1/FinalStageSVM2.npy')
FinalStageSVM3 = joblib.load('FinalEnsemble1/FinalStageSVM3.npy')
FinalStageSVM4 = joblib.load('FinalEnsemble1/FinalStageSVM4.npy')
FinalStageSVM5 = joblib.load('FinalEnsemble1/FinalStageSVM5.npy')
FinalStageSVM6 = joblib.load('FinalEnsemble1/FinalStageSVM6.npy')
FinalStageSVM7 = joblib.load('FinalEnsemble1/FinalStageSVM7.npy')
FinalStageSVM8 = joblib.load('FinalEnsemble1/FinalStageSVM8.npy')
FinalStageSVM9 = joblib.load('FinalEnsemble1/FinalStageSVM9.npy')
FinalStageSVM10 = joblib.load('FinalEnsemble1/FinalStageSVM10.npy')
FinalStageSVM11 = joblib.load('FinalEnsemble1/FinalStageSVM11.npy')
FinalStageSVM12 = joblib.load('FinalEnsemble1/FinalStageSVM12.npy')
FinalStageSVM13 = joblib.load('FinalEnsemble1/FinalStageSVM13.npy')
FinalStageSVM14 = joblib.load('FinalEnsemble1/FinalStageSVM14.npy')
FinalStageSVM15 = joblib.load('FinalEnsemble1/FinalStageSVM15.npy')
FinalStageSVM16 = joblib.load('FinalEnsemble1/FinalStageSVM16.npy')


def SequentialModel2(input):
    y_stage1_proba = SVM_stage1.predict_proba(input)
    y_stage1_proba = y_stage1_proba.tolist()
    y_stage1_p = max(y_stage1_proba)
    y_stage1 = SVM_stage1.predict(input)

    # Stage 1
    if y_stage1 == 'Group1':
        y_stage2 = SVM1_stage2.predict(input)
        y_stage2_proba = SVM1_stage2.predict_proba(input)
        y_stage2_proba = y_stage2_proba.tolist()
        y_stage2_p = max(y_stage2_proba)

        # Stage 2
        if y_stage2 == 'Group1':
            y_stage3 = SVM1_stage3.predict(input)

            # Stage 3
            if y_stage3 == 'Group1':
                y_stage4 = SVM1_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM1.predict(input)
                else:
                    # Final
                    output = FinalStageSVM2.predict(input)
            else:
                y_stage4 = SVM2_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM3.predict(input)
                else:
                    # Final
                    output = FinalStageSVM4.predict(input)
        else:
            y_stage3 = SVM2_stage3.predict(input)

            # Stage 3
            if y_stage3 == 'Group1':
                y_stage4 = SVM3_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM5.predict(input)
                else:
                    # Final
                    output = FinalStageSVM6.predict(input)
            else:
                y_stage4 = SVM4_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM7.predict(input)
                else:
                    # Final
                    output = FinalStageSVM8.predict(input)

    else:
        y_stage2 = SVM2_stage2.predict(input)

        # Stage 2
        if y_stage2 == 'Group1':
            y_stage3 = SVM3_stage3.predict(input)

            # Stage 3
            if y_stage3 == 'Group1':
                y_stage4 = SVM5_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM9.predict(input)
                else:
                    # Final
                    output = FinalStageSVM10.predict(input)
            else:
                y_stage4 = SVM6_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM11.predict(input)
                else:
                    # Final
                    output = FinalStageSVM12.predict(input)
        else:
            y_stage3 = SVM4_stage3.predict(input)

            # Stage 3
            if y_stage3 == 'Group1':
                y_stage4 = SVM7_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM13.predict(input)
                else:
                    # Final
                    output = FinalStageSVM14.predict(input)
            else:
                y_stage4 = SVM8_stage4.predict(input)

                # Stage 4
                if y_stage4 == 'Group1':

                    # Final stage
                    output = FinalStageSVM15.predict(input)
                else:
                    # Final
                    output = FinalStageSVM16.predict(input)
    return output