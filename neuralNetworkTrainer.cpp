#include "opencv2/core/core_c.h"
#include "opencv2/ml/ml.hpp"
#include "constants.h"
#include <cstdio>
#include <vector>

// execute this line
// ./TrainNetwork -data ./data/neural/letters.txt -mlp -save ./data/neural/neuralData.xml


static void help()
{
    printf("\n=== Neural Network trainer for sign language backproj ===\n");
}

// This function reads data and responses from the file <filename>

/*
Fetch backproj data & parse it to form a neural network
*/
static int read_num_class_data( const char* filename, int var_count, CvMat** data, CvMat** responses )
{
    printf("Reading backproj data...\n");
    const int M = 1024;
    FILE* f = fopen( filename, "rt" );
    CvMemStorage* storage;
    CvSeq* seq;
    char buf[M+2];
    float* el_ptr;
    CvSeqReader reader;
    int i, j;

    if( !f )
        return 0;

    el_ptr = new float[var_count+1];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq( 0, sizeof(*seq), (var_count+1)*sizeof(float), storage );

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        el_ptr[0] = buf[0];
        ptr = buf+2;
        for( i = 1; i <= var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", el_ptr + i, &n );
            ptr += n + 1;
        }
        if( i <= var_count )
            break;
        cvSeqPush( seq, el_ptr );
    }
    fclose(f);

    *data = cvCreateMat( seq->total, var_count, CV_32F );
    *responses = cvCreateMat( seq->total, 1, CV_32F );

    cvStartReadSeq( seq, &reader );

    for( i = 0; i < seq->total; i++ )
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = data[0]->data.fl + var_count*i;
        float* dr = responses[0]->data.fl + i;

        for( j = 0; j < var_count; j++ )
            ddata[j] = sdata[j];
        *dr = sdata[-1];
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    cvReleaseMemStorage( &storage );
    delete[] el_ptr;
    return 1;
}

static int build_mlp_classifier( char* data_filename, char* filename_to_save, char* filename_to_load )
{
    printf("Building classifier...\n");
    const int class_count = NUMBER_OF_LETTERS_IN_NEURAL_NETWORK; // Alphabet : 26 letters
    CvMat* data = 0;
    CvMat train_data;
    CvMat* responses = 0;
    CvMat* mlp_response = 0;

    // 256 au lieu de 16
    int ok = read_num_class_data( data_filename, 256, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int i, j;
    double train_hr = 0, test_hr = 0;
    CvANN_MLP mlp;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load MLP classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        mlp.load( filename_to_load );
        ntrain_samples = 0;
        if( !mlp.get_layer_count() )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", filename_to_load );
    }
    else
    {
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        //
        // MLP does not support categorical variables by explicitly.
        // So, instead of the output class label, we will use
        // a binary vector of <class_count> components for training and,
        // therefore, MLP will give us a vector of "probabilities" at the
        // prediction stage
        //
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        CvMat* new_responses = cvCreateMat( ntrain_samples, class_count, CV_32F );

        // 1. unroll the responses
        printf( "Unrolling the responses...\n");
        for( i = 0; i < ntrain_samples; i++ )
        {
            int cls_label = cvRound(responses->data.fl[i]) - 'A';
            float* bit_vec = (float*)(new_responses->data.ptr + i*new_responses->step);
            for( j = 0; j < class_count; j++ )
                bit_vec[j] = 0.f;
            bit_vec[cls_label] = 1.f;
        }
        cvGetRows( data, &train_data, 0, ntrain_samples );

        // 2. train classifier
        int layer_sz[] = { data->cols, 4, class_count }; // TODO: j'ai mis 4 au lieu de 100,100
        CvMat layer_sizes =
            cvMat( 1, (int)(sizeof(layer_sz)/sizeof(layer_sz[0])), CV_32S, layer_sz );
        mlp.create( &layer_sizes );
        printf( "Training the classifier (may take a few minutes)...\n");

#if 1
        int method = CvANN_MLP_TrainParams::BACKPROP;
        double method_param = 0.001;
        int max_iter = 300;
#else
        int method = CvANN_MLP_TrainParams::RPROP;
        double method_param = 0.1;
        int max_iter = 1000;
#endif

        mlp.train( &train_data, new_responses, 0, 0,
            CvANN_MLP_TrainParams(cvTermCriteria(CV_TERMCRIT_ITER,max_iter,0.01),
                                  method, method_param));
        cvReleaseMat( &new_responses );
        printf("\n");
    }

    mlp_response = cvCreateMat( 1, class_count, CV_32F );

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        int best_class;
        CvMat sample;
        cvGetRow( data, &sample, i );
        CvPoint max_loc = {0,0};
        mlp.predict( &sample, mlp_response );
        cvMinMaxLoc( mlp_response, 0, 0, 0, &max_loc, 0 );
        best_class = max_loc.x + 'A';

        int r = fabs((double)best_class - responses->data.fl[i]) < FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    // Save classifier to file if needed
    if( filename_to_save )
        mlp.save( filename_to_save );

    cvReleaseMat( &mlp_response );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    printf("Classifier built\n");
    return 0;
}

int main( int argc, char *argv[] )
{
    help();
    char* filename_to_save = 0;
    char* filename_to_load = 0;
    char default_data_filename[] = "./letter-recognition.data";
    char* data_filename = default_data_filename;
    int method = 0;

    int i;
    for( i = 1; i < argc; i++ )
    {
        if( strcmp(argv[i],"-data") == 0 ) // flag "-data letter_recognition.xml"
        {
            i++;
            data_filename = argv[i];
        }
        else if( strcmp(argv[i],"-save") == 0 ) // flag "-save filename.xml"
        {
            i++;
            filename_to_save = argv[i];
        }
        else if( strcmp(argv[i],"-load") == 0) // flag "-load filename.xml"
        {
            i++;
            filename_to_load = argv[i];
        }
        else if( strcmp(argv[i],"-boost") == 0)
        {
            method = 1;
        }
        else if( strcmp(argv[i],"-mlp") == 0 )
        {
            method = 2;
        }
        else if ( strcmp(argv[i], "-knearest") == 0)
    {
        method = 3;
    }
    else if ( strcmp(argv[i], "-nbayes") == 0)
    {
        method = 4;
    }
    else if ( strcmp(argv[i], "-svm") == 0)
    {
        method = 5;
    }
        else
            break;
    }

    if( i < argc ||
        (
        method == 2 ?
        build_mlp_classifier( data_filename, filename_to_save, filename_to_load ) :

        -1) < 0)
    {
    }
    return 0;
}
