import base64
import streamlit as st
import pandas as pd
import numpy as np
import Bio
from Bio import Align

import matplotlib.pyplot as plt
import os
import warnings
from PIL import Image
from pympler.util.bottle import app
from streamlit_option_menu import option_menu
import webbrowser
import sys
from streamlit.components.v1 import html
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def getKmers(sequence, size=6):
    return [sequence[x:x + size].lower() for x in range(len(sequence) - size + 1)]


def RunTests(dataframe, human_data):
    testseq = dataframe

    testseq['K-mers'] = testseq.apply(lambda x: getKmers(x['sequence']), axis=1)
    testseq = testseq.drop('sequence', axis=1)

    human_data['K-mers'] = human_data.apply(lambda x: getKmers(x['sequence']), axis=1)
    human_data = human_data.drop('sequence', axis=1)

    human_texts = list(human_data['K-mers'])
    for item in range(len(human_texts)):
        human_texts[item] = ' '.join(human_texts[item])
    y_data = human_data.iloc[:, 0].values

    test_texts = list(testseq['K-mers'])
    for item in range(len(test_texts)):
        test_texts[item] = ' '.join(test_texts[item])

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(ngram_range=(4, 4))
    X = cv.fit_transform(human_texts)
    T = cv.transform(test_texts)

    # Splitting the human dataset into the training set and test set
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)
    ### Multinomial Naive Bayes Classifier ###
    # The alpha parameter was determined by grid search previously
    from sklearn.naive_bayes import MultinomialNB

    classifier = MultinomialNB(alpha=0.1)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(T)

    a = pd.DataFrame(y_pred)
    a.columns = ["Classes"]

    # Our Class Dicitonary
    class_dictionary = {
        0: "G Protein-coupled Receptors",
        1: "Tyrosine Kinase",
        2: "Tyrosine Phosphatases",
        3: "Synthetase",
        4: "Synthase",
        5: "Ion channels",
        6: "Transcription Factor"
    }

    # mapping on the dicitonary
    a["Classes"] = [*map(class_dictionary.get, a["Classes"])]

    tesClone = pd.concat([dataframe, a], axis=1)

    tesClone.to_csv('sample.csv', index=False)
    return tesClone


if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False


def callback():
    st.session_state.button_clicked = False


warnings.simplefilter(action='ignore', category=FutureWarning)

image = Image.open(r'C:\Users\TC\Documents\project\gene.png')

st.set_page_config(
    page_title="DNA Classification and Alignment",
    page_icon=image,
    initial_sidebar_state='collapsed',
)

st.markdown(
    """
    <style>
         .main {
            text-align: center;
         }
        .st-al:hover p{
            color:#fff;
        }
        /* Text area */
        .st-d0{
        }
        textarea[type="textarea"]{
            cursor: text;
            font: bold 28px monospace;
            color: tan;
            resize: none;
            height: 220px;
        }
        textarea[type="textarea"]::selection{
            color: #222;
            background-color: orangered;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Execute your app

# Starting Sequence Alignments
DNA = ["A", "G", "C", "T"]


@st.cache
def sequence_checker(seq):
    for i in seq:
        if i not in DNA:
            return False
    return True


@st.cache
def global_sequence_alignment(seq1, seq2, match=1, mismatch=-1, gap=-2, dna_seq=True):
    if dna_seq:
        if sequence_checker(seq1) and sequence_checker(seq2):
            # create Matrices
            main_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1), int)
            match_checker_matrix = np.zeros((len(seq1), len(seq2)), int)

            # Providing The Score Penalty
            match_reward = match
            mismatch_penalty = mismatch
            gap_penalty = gap

            # Fill The Match Checker Matrix According To Match or Mismatch
            for i in range(len(seq1)):
                for j in range(len(seq2)):
                    if seq1[i] == seq2[j]:
                        match_checker_matrix[i][j] = match_reward
                    else:
                        match_checker_matrix[i][j] = mismatch_penalty

            # Filling Up The Main Matrix Using Needleman_Wunsch Algorithm
            # STEP 1: Initialization

            # main_matrix[row][column]
            # Looping in Rows
            for i in range(len(seq1) + 1):
                main_matrix[i][0] = i * gap_penalty

            # Looping in Columns
            for j in range(len(seq2) + 1):
                main_matrix[0][j] = j * gap_penalty

            # STEP 2: Filling The Matrix
            for i in range(1, len(seq1) + 1):
                for j in range(1, len(seq2) + 1):
                    calc_diagonal = main_matrix[i - 1][j - 1] + match_checker_matrix[i - 1][j - 1]
                    calc_bottom = main_matrix[i - 1][j] + gap_penalty
                    calc_right = main_matrix[i][j - 1] + gap_penalty
                    main_matrix[i][j] = max(calc_diagonal, calc_right, calc_bottom)

            # STEP 3: Tracebacking
            aligned_1 = ""
            aligned_2 = ""

            ti = len(seq1)
            tj = len(seq2)

            # alignment Score Is The Last Cell In Matrix
            alignment_score = main_matrix[ti][tj]

            traceback_node = []
            traceback_position = []
            while ti > 0 or tj > 0:
                if main_matrix[ti][tj] == main_matrix[ti - 1][tj - 1] + match_checker_matrix[ti - 1][tj - 1]:
                    aligned_1 = seq1[ti - 1] + aligned_1
                    aligned_2 = seq2[tj - 1] + aligned_2

                    traceback_node.append(main_matrix[ti][tj])
                    traceback_position.append(tuple((ti, tj)))
                    ti -= 1
                    tj -= 1

                elif ti > 0 and main_matrix[ti][tj] == main_matrix[ti - 1][tj] + gap_penalty:
                    aligned_1 = seq1[ti - 1] + aligned_1
                    aligned_2 = "-" + aligned_2

                    traceback_node.append(main_matrix[ti][tj])
                    traceback_position.append(tuple((ti, tj)))
                    ti -= 1

                else:
                    aligned_1 = "-" + aligned_1
                    aligned_2 = seq2[tj - 1] + aligned_2

                    traceback_node.append(main_matrix[ti][tj])
                    traceback_position.append(tuple((ti, tj)))

                    tj -= 1

            return aligned_1, aligned_2, main_matrix, alignment_score, traceback_node, traceback_position
        else:
            return False
    else:
        # create Matrices
        main_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1), int)
        match_checker_matrix = np.zeros((len(seq1), len(seq2)), int)

        # Providing The Score Penalty
        match_reward = match
        mismatch_penalty = mismatch
        gap_penalty = gap

        # Fill The Match Checker Matrix According To Match or Mismatch
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    match_checker_matrix[i][j] = match_reward
                else:
                    match_checker_matrix[i][j] = mismatch_penalty

        # Filling Up The Main Matrix Using Needleman_Wunsch Algorithm
        # STEP 1: Initialization

        # main_matrix[row][column]
        # Looping in Rows
        for i in range(len(seq1) + 1):
            main_matrix[i][0] = i * gap_penalty

        # Looping in Columns
        for j in range(len(seq2) + 1):
            main_matrix[0][j] = j * gap_penalty

        # STEP 2: Filling The Matrix
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                calc_diagonal = main_matrix[i - 1][j - 1] + match_checker_matrix[i - 1][j - 1]
                calc_bottom = main_matrix[i - 1][j] + gap_penalty
                calc_right = main_matrix[i][j - 1] + gap_penalty
                main_matrix[i][j] = max(calc_diagonal, calc_right, calc_bottom)

        # STEP 3: Tracebacking
        aligned_1 = ""
        aligned_2 = ""

        ti = len(seq1)
        tj = len(seq2)

        # alignment Score Is The Last Cell In Matrix
        alignment_score = main_matrix[ti][tj]

        traceback_node = []
        traceback_position = []
        while ti > 0 or tj > 0:
            if main_matrix[ti][tj] == main_matrix[ti - 1][tj - 1] + match_checker_matrix[ti - 1][tj - 1]:
                aligned_1 = seq1[ti - 1] + aligned_1
                aligned_2 = seq2[tj - 1] + aligned_2

                traceback_node.append(main_matrix[ti][tj])
                traceback_position.append(tuple((ti, tj)))
                ti -= 1
                tj -= 1

            elif ti > 0 and main_matrix[ti][tj] == main_matrix[ti - 1][tj] + gap_penalty:
                aligned_1 = seq1[ti - 1] + aligned_1
                aligned_2 = "-" + aligned_2

                traceback_node.append(main_matrix[ti][tj])
                traceback_position.append(tuple((ti, tj)))
                ti -= 1

            else:
                aligned_1 = "-" + aligned_1
                aligned_2 = seq2[tj - 1] + aligned_2

                traceback_node.append(main_matrix[ti][tj])
                traceback_position.append(tuple((ti, tj)))

                tj -= 1

        return aligned_1, aligned_2, main_matrix, alignment_score, traceback_node, traceback_position


aligner = Align.PairwiseAligner()


@st.cache
def get_all_possible_alignmets(s1, s2, match, mismatch, gap, mode):
    aligner.mode = mode
    aligner.match = match
    aligner.mismatch = mismatch
    aligner.gap_score = gap
    alignments_list = aligner.align(s1, s2)

    all_alignment = []
    for i in alignments_list:
        all_alignment.append(str(i))
    return all_alignment


@st.cache
def local_sequence_alignment(s1, s2, match=1, mismatch=-1, gap=-2, dna_seq=True):
    if dna_seq:
        if sequence_checker(s1) and sequence_checker(s2):
            matrix = np.zeros((len(s1) + 2, len(s2) + 2), dtype=int)

            rowN, columN = matrix.shape

            for i in range(2, rowN):
                matrix[i][0] = ord(s1[i - 2])

            for i in range(2, columN):
                matrix[0][i] = ord(s2[i - 2])

            for i in range(2, rowN):
                for j in range(2, columN):
                    left = matrix[i][j - 1] + gap
                    up = matrix[i - 1][j] + gap
                    diag = matrix[i - 1][j - 1]

                    if (matrix[i][0] == matrix[0][j]):
                        diag += match
                    else:
                        diag += mismatch

                    Maxi = max([left, up, diag])
                    if (Maxi >= 0):
                        matrix[i][j] = Maxi
                    else:
                        matrix[i][j] = 0

            matrix_del = np.delete(matrix, 0, 0)
            matrix_del = np.delete(matrix_del, 0, 1)

            maxStart = max(list(map(max, matrix_del)))

            result = np.where(matrix == maxStart)

            i, j = result[0][0], result[1][0]
            score = matrix[i][j]

            align1 = ''
            align2 = ''
            matMis = ''
            path = []
            path_pos = []

            while matrix[i][j] != 0:

                current = matrix[i][j]
                path.append(current)
                path_pos.append(tuple((i - 1, j - 1)))

                if matrix[i][0] == matrix[0][j]:
                    align1 += chr(matrix[i][0])
                    align2 += chr(matrix[0][j])
                    matMis += '*'
                    i, j = i - 1, j - 1
                else:
                    left = matrix[i][j - 1]
                    up = matrix[i - 1][j]
                    diag = matrix[i - 1][j - 1]

                    maxi = max([left, up, diag])

                    if i == 1:
                        maxi = left

                    if maxi == left:
                        align1 += '-'
                        align2 += chr(matrix[0][j])
                        matMis += ' '
                        j -= 1

                    elif maxi == up:

                        align1 += chr(matrix[i][0])
                        align2 += '-'
                        matMis += ' '
                        i -= 1

                    else:
                        align1 += chr(matrix[i][0])
                        align2 += chr(matrix[0][j])
                        matMis += '|'
                        i, j = i - 1, j - 1

            path.append(0)

            return align1[::-1], align2[::-1], matrix_del, score, path_pos[0], path, path_pos
        else:
            return False
    else:
        matrix = np.zeros((len(s1) + 2, len(s2) + 2), dtype=int)

        rowN, columN = matrix.shape

        for i in range(2, rowN):
            matrix[i][0] = ord(s1[i - 2])

        for i in range(2, columN):
            matrix[0][i] = ord(s2[i - 2])

        for i in range(2, rowN):
            for j in range(2, columN):
                left = matrix[i][j - 1] + gap
                up = matrix[i - 1][j] + gap
                diag = matrix[i - 1][j - 1]

                if (matrix[i][0] == matrix[0][j]):
                    diag += match
                else:
                    diag += mismatch

                Maxi = max([left, up, diag])
                if (Maxi >= 0):
                    matrix[i][j] = Maxi
                else:
                    matrix[i][j] = 0

        matrix_del = np.delete(matrix, 0, 0)
        matrix_del = np.delete(matrix_del, 0, 1)

        maxStart = max(list(map(max, matrix_del)))

        result = np.where(matrix == maxStart)

        i, j = result[0][0], result[1][0]
        score = matrix[i][j]

        align1 = ''
        align2 = ''
        matMis = ''
        path = []
        path_pos = []

        while matrix[i][j] != 0:

            current = matrix[i][j]
            path.append(current)
            path_pos.append(tuple((i - 1, j - 1)))

            if matrix[i][0] == matrix[0][j]:
                align1 += chr(matrix[i][0])
                align2 += chr(matrix[0][j])
                matMis += '*'
                i, j = i - 1, j - 1
            else:
                left = matrix[i][j - 1]
                up = matrix[i - 1][j]
                diag = matrix[i - 1][j - 1]

                maxi = max([left, up, diag])

                if i == 1:
                    maxi = left

                if maxi == left:
                    align1 += '-'
                    align2 += chr(matrix[0][j])
                    matMis += ' '
                    j -= 1

                elif maxi == up:

                    align1 += chr(matrix[i][0])
                    align2 += '-'
                    matMis += ' '
                    i -= 1

                else:
                    align1 += chr(matrix[i][0])
                    align2 += chr(matrix[0][j])
                    matMis += '|'
                    i, j = i - 1, j - 1

        path.append(0)
        if len(align1[::-1]) > 0 or len(align2[::-1]) > 0:
            return align1[::-1], align2[::-1], matrix_del, score, path_pos[0], path, path_pos
        else:
            return ""


# End Sequence Alignment
header = st.container()
datset = st.container()
features = st.container()
model_training = st.container()
with st.sidebar:
    st.title("DNA Classification and Alignment")
    selected = option_menu("Main menu", ["Home", "Classifier", 'Global Sequence Alignment', 'Local Sequence Alignment'],
                           menu_icon="cast", default_index=2)

    if selected == "Home":
        with datset:
            st.title('Welcome to our Awesome Project !')
            col1, col2, col3 = st.columns([1, 6, 1])
            with col2:
                image = Image.open(r'C:\Users\TC\Documents\project\gene.png')
                st.image(image, width=500)

    if selected == 'Classifier':
        def mode_act(dat, tnum):

            dat['K-mers'] = dat.apply(lambda x: getKmers(x['sequence']), axis=1)
            dat = dat.drop('sequence', axis=1)

            human_texts = list(dat['K-mers'])
            for item in range(len(human_texts)):
                human_texts[item] = ' '.join(human_texts[item])

            y_data = dat.iloc[:, 0].values

            cv = CountVectorizer(ngram_range=(4, 4))
            X = cv.fit_transform(human_texts)

            accuracy_tests = []
            for i in range(tnum):
                X_train, X_test, y_train, y_test = train_test_split(X,
                                                                    y_data,
                                                                    test_size=0.20)
                classifier = MultinomialNB(alpha=0.1)
                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)
                # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

                num_test = X_test.shape[0]

                # log_clf_preds = log_clf.predict(X_test)
                log_clf_accuracy = (y_pred == y_test)
                test_accurecy = format(np.sum(log_clf_accuracy) / num_test, '.2%')
                accuracy_tests.append(test_accurecy)

            display_col.subheader("Minimum Test Accuracy")
            display_col.write(min(accuracy_tests))
            display_col.subheader("Maximum Test Accuracy")
            display_col.write(max(accuracy_tests))


        @st.cache(allow_output_mutation=True)
        def get_data(filename):
            data = pd.read_table(filename)
            return data


        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False)


        def show(file):
            if st.button("Display Result"):
                if file is not None:
                    st.dataframe(file)


        with header:
            st.title('DNA Sequence Classifier!')

            c1, c2, c3 = st.columns((5, 5, 5))
            with c2:
                image = Image.open(r'C:\Users\TC\Documents\project\classTree.jpeg')
                st.image(image, width=200)

            st.text(F'In This Project We are Looking Forward to Providing a Simple Tool\n'
                    'for DNA Sequencing Classifier Using Machine Learning')

        with datset:

            st.header('Human Dataset')
            st.text('We found this data set at Kaggle.com\n')
            c1, c2, c3 = st.columns([1, 3, 1])
            with c2:
                train_data = get_data(r'human_data.txt')
                st.dataframe(train_data)

            w1, w2, w3 = st.columns([1, 3, 1])

            # with w2:
            # display_col, selection_col = st.columns(2)
            st.write("")
            st.write("")
            st.write("")
            st.subheader("If we have a look at class balance we can see we have relatively balanced dataset.".title())
            st.bar_chart(train_data['class'].value_counts().sort_index())
            # train_data['class'].value_counts().sort_index().plot.bar()

        with features:
            st.header('The features we created')
            st.text('Let\'s take a look into the features we generated.')
            st.markdown(
                '* Classify DNA :This is the main feature of or project that configures the gene family of your DNA')
            st.markdown(
                '* Test Model Accuracy:This feature was made to enable the user to run tests on the model to realise '
                'the expected range of model accuracy')

        with model_training:
            st.header('Time to train the model !')
            st.text('In this section you can select the hyper parameters !')

            selection_col, display_col = st.columns(2)
            human_data = get_data(r'human_data.txt')

            if selected == 'Classifier':
                c1, c2, c3 = st.columns([1, 3, 1])
                with c2:

                    st.subheader('Model Accuracy')
                    tnum = st.slider('Choose the number of test to run on the model', min_value=1,
                                     max_value=10,
                                     value=5,
                                     step=1)

                    if st.button("Run Tests"):
                        mode_act(human_data, tnum)

                st.subheader('Time to put the model in actual use!')

                try:
                    uploaded_file = st.file_uploader("Choose a file")
                    if uploaded_file is not None:
                        dataframe = pd.read_csv(uploaded_file)
                        tesClone = RunTests(dataframe, human_data)

                        show(tesClone)

                        csv = convert_df(tesClone)
                        # csv.drop(columns=csv.columns[0],axis=1, inplace=True)

                        st.download_button(
                            label="Download result as CSV",
                            data=csv,
                            file_name='large_df.csv',
                            mime='text/csv',
                        )
                        st.success('Model worked successfully')


                except:
                    st.error('''Oops ! error occurred\n
                    Make sure .csv file is selected ,and your data is valid for this model\n
                    \"Consider dataset above is a sample of valid data\"''')

            else:
                st.error(''' Opps! Some thing went wrong''')

    elif selected == 'Global Sequence Alignment':
        with header:
            st.title('Global Sequence Alignment'.title())
            c1, c2, c3 = st.columns((5, 2, 5))
            with c2:
                image = Image.open(r'C:\Users\TC\Documents\project\seq.png')
                st.image(image, caption='', width=100)
            st.text(
                F'The purpose of this app is to get the alignment matrix, alignment score,\nalignment sequences,'
                F'all possible alignments.\n '
                F'based on user defined Match, Mismatch and Gap Scores'.title())

        with datset:
            def check_space_and_digit(seq):
                if len(seq) == 0:
                    return False
                new_sew = ""
                for l in seq:
                    if l.isalpha():
                        new_sew = new_sew + l
                    else:
                        new_sew = new_sew + ""
                return new_sew


            def update_first():
                if len(st.session_state.seq1) != 0:
                    st.session_state.seq1 = check_space_and_digit(st.session_state.seq1).upper()


            def update_second():
                if len(st.session_state.seq2) != 0:
                    st.session_state.seq2 = check_space_and_digit(st.session_state.seq2).upper()


            just_dna = st.checkbox("Only DNA Sequence Alignment", value=True)
            first_seq = st.text_input(label="First Sequence", disabled=False, value="ACCG", key="seq1",
                                      on_change=update_first).upper().strip()
            second_seq = st.text_input(label="Second Sequence", disabled=False, value="ACGC", key="seq2",
                                       on_change=update_second).upper().strip()

            if len(first_seq) == 0 or len(second_seq) ==0:
                st.warning("You have to enter sequence in fields".upper())

            else:

                [c1, c2, c3] = st.columns(3)

                with c1:
                    match_reward = st.number_input('Match', min_value=1, max_value=20, value=1, step=1)
                with c2:
                    mismatch_penalty = st.number_input('Mismatch', min_value=-20, max_value=20, value=-1, step=1)
                with c3:
                    gap_penalty = st.number_input('Gap', min_value=-20, max_value=20, value=-2, step=1)

                global_alignment_results = global_sequence_alignment(first_seq, second_seq, match_reward,
                                                                     mismatch_penalty,
                                                                     gap_penalty, just_dna)

                if global_alignment_results == False:
                    st.error("Input Fields Must Consists of Base Pairs [A, C, G, T]")

                else:

                    seq1_alignment = global_alignment_results[0]
                    seq2_alignment = global_alignment_results[1]
                    main_matrix = global_alignment_results[2]
                    alignment_score = global_alignment_results[3]
                    traceback_position = global_alignment_results[5]
                    traceback_node = global_alignment_results[4]

                    st.text(traceback_node)
                    st.text(traceback_position)

                    num_row = main_matrix.shape[0]
                    num_col = main_matrix.shape[1]

                    # Filling First Row With Sequence
                    cols = st.columns(len(second_seq) + 2)
                    for i in range(len(second_seq) + 2):
                        if i == 0:
                            cols[i].success(f'D')
                        elif i == 1:
                            cols[i].success(f'•')
                        else:
                            cols[i].success(f'{second_seq[i - 2]}')

                    cols = st.columns(num_col + 1)

                    # Filling All Matrix With main matrix
                    for r in range(num_row + 3):
                        for c in range(num_col + 1):
                            if c == 0 and r == 0:
                                cols[c].success(f"•")

                            elif c == 0 and 0 < r <= len(first_seq):
                                cols[c].success(f"{first_seq[r - 1]}")
                            else:
                                if c == num_col and r == num_row - 1:
                                    cols[c].error(f'{main_matrix[r][c - 1]}')

                                elif (tuple((r, c - 1)) in traceback_position) or (c - 1 == 0 and r == 0):
                                    cols[c].error(f'{main_matrix[r][c - 1]}')

                                elif r < num_row:
                                    cols[c].info(f'{main_matrix[r][c - 1]}')

                    breaker = []
                    for i in range(len(seq1_alignment)):
                        if seq1_alignment[i] != "-" and seq2_alignment[i] != "-":
                            if seq1_alignment[i] == seq2_alignment[i]:
                                breaker.append("|")
                            else:
                                breaker.append("•")
                        else:
                            breaker.append(" ")
                    if st.button('Get Sequence Alignments'):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            s = st.text_area("Sequence Alignment",
                                             value=f'{" ".join(seq1_alignment)}\n{" ".join(breaker)}\n{" ".join(seq2_alignment)}',
                                             disabled=True)
                        with col2:
                            st.metric("Score", f"{alignment_score}")

                    if st.button('Get All Possible Alignments'):
                        aligner.mode = "global"
                        aligner.match = match_reward
                        aligner.mismatch = mismatch_penalty
                        aligner.gap_score = gap_penalty
                        alignments_list = aligner.align(first_seq, second_seq)

                        all_alignment = []
                        for i in alignments_list:
                            all_alignment.append(str(i))

                        st.text_area(label="All Possible Alignments", value=str("\n".join(all_alignment)), disabled=True)
                        st.success("Global Alignment Applied Successfully")

    elif selected == 'Local Sequence Alignment':
        with header:

            st.title('Local Sequence Alignment'.title())
            c1, c2, c3 = st.columns((5, 2, 5))
            with c2:
                image = Image.open(r'C:\Users\TC\Documents\project\localDna.png')
                st.image(image, caption='', width=100)
            st.text(
                F'The purpose of this app is to get the alignment matrix, alignment score,\nalignment sequences,'
                F'all possible alignments.\n '
                F'based on user defined Match, Mismatch and Gap Scores'.title())

        with datset:

            def check_space_and_digit(seq):
                flag = False
                new_sew = ""
                for l in seq:
                    if l.isalpha():
                        new_sew = new_sew + l
                    else:
                        new_sew = new_sew + ""
                return new_sew


            def update_first():
                if len(st.session_state.s1) != 0:
                    st.session_state.s1 = check_space_and_digit(st.session_state.s1).upper()


            def update_second():
                if len(st.session_state.s2) != 0:
                    st.session_state.s2 = check_space_and_digit(st.session_state.s2).upper()


            just_dna = st.checkbox("Only DNA Sequence Alignment", value=True)

            first_seq = st.text_input(label="First Sequence", disabled=False, value="ACGTGCCG", key="s1",
                                      on_change=update_first).upper().strip()
            second_seq = st.text_input(label="Second Sequence", disabled=False, value="ACGC", key="s2",
                                       on_change=update_second).upper().strip()



            if len(first_seq) == 0 or len(second_seq) ==0:
                st.warning("You have to enter sequence in fields".upper())
            else:
                [c1, c2, c3] = st.columns(3)
                with c1:
                    match_reward = st.number_input('Match', min_value=1, max_value=20, value=1, step=1)
                with c2:
                    mismatch_penalty = st.number_input('Mismatch', min_value=-20, max_value=20, value=-1, step=1)
                with c3:
                    gap_penalty = st.number_input('Gap', min_value=-20, max_value=20, value=-2, step=1)

                local_alignment_results = local_sequence_alignment(first_seq, second_seq, match_reward, mismatch_penalty,
                                                                   gap_penalty, just_dna)

                if local_alignment_results == False:
                    st.error("Input Fields Must Consists of Base Pairs [A, C, G, T]")

                elif local_alignment_results != "":
                    seq1_alignment = local_alignment_results[0]
                    seq2_alignment = local_alignment_results[1]
                    main_matrix = local_alignment_results[2]
                    alignment_score = local_alignment_results[3]
                    score_position = local_alignment_results[4]
                    traceback_node = local_alignment_results[5]
                    traceback_position = local_alignment_results[6]

                    st.text(traceback_node)
                    st.text(traceback_position)

                    num_row = main_matrix.shape[0]
                    num_col = main_matrix.shape[1]

                    # Filling First Row With Sequence
                    cols = st.columns(len(second_seq) + 2)
                    for i in range(len(second_seq) + 2):
                        if i == 0:
                            cols[i].success(f'D')
                        elif i == 1:
                            cols[i].success(f'#')
                        else:
                            cols[i].success(f'{second_seq[i - 2]}')

                    cols = st.columns(num_col + 1)

                    # Filling All Matrix With main matrix
                    for r in range(num_row + 3):
                        for c in range(num_col + 1):
                            if c == 0 and (r == 0):
                                cols[c].success(f"#")

                            elif c == 0 and 0 < r <= len(first_seq):
                                cols[c].success(f"{first_seq[r - 1]}")

                            else:
                                if r == score_position[0] and c - 1 == score_position[1]:
                                    cols[c].error(f'{main_matrix[r][c - 1]}')

                                elif tuple((r, c - 1)) in traceback_position:
                                    cols[c].error(f'{main_matrix[r][c - 1]}')

                                elif r < num_row:
                                    cols[c].info(f'{main_matrix[r][c - 1]}')

                    breaker = []
                    for i in range(len(seq1_alignment)):
                        if seq1_alignment[i] != "-" and seq2_alignment[i] != "-":
                            if seq1_alignment[i] == seq2_alignment[i]:
                                breaker.append("|")
                            else:
                                breaker.append("•")
                        else:
                            breaker.append(" ")

                    if st.button('Get Sequence Alignments'):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            s = st.text_area("Sequence Alignment",
                                             value=f'{" ".join(seq1_alignment)}\n{" ".join(breaker)}\n{" ".join(seq2_alignment)}',
                                             disabled=True)
                        with col2:
                            st.metric("Score", f"{alignment_score}")

                    if st.button('Get All Possible Alignments'):
                        aligner.mode = "local"
                        aligner.match = match_reward
                        aligner.mismatch = mismatch_penalty
                        aligner.gap_score = gap_penalty
                        alignments_list = aligner.align(first_seq, second_seq)

                        all_alignment = []
                        for i in alignments_list:
                            all_alignment.append(str(i))

                        st.text_area(label="All Possible Alignments", value=str("\n".join(all_alignment)), disabled=True)
                        st.success("Local Alignment Applied Successfully")

                else:
                    st.error("No Local Sequence Alignments Available.. \nNo matched characters!")