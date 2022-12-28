package com.example.irismlapp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.example.irismlapp.ml.Iris;
import com.example.irismlapp.ml.IrisModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {

    EditText n1;
    EditText n2;
    EditText n3;
    EditText n4;
    TextView test;
    Button predict;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        n1 = findViewById(R.id.n1);
        n2 = findViewById(R.id.n2);
        n3 = findViewById(R.id.n3);
        n4 = findViewById(R.id.n4);

        predict = findViewById(R.id.predictBtn);

        test = findViewById(R.id.test);


    }

    public void predict(View v){
        float v1 = Float.valueOf(n1.getText().toString());
        float v2 = Float.valueOf(n2.getText().toString());
        float v3 = Float.valueOf(n3.getText().toString());
        float v4 = Float.valueOf(n4.getText().toString());

        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(16);
        byteBuffer.putFloat(v1);
        byteBuffer.putFloat(v2);
        byteBuffer.putFloat(v3);
        byteBuffer.putFloat(v4);

        try {
            IrisModel model = IrisModel.newInstance(this);

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            IrisModel.Outputs outputs = model.process(inputFeature0);
            float[] outputFeature0 = outputs.getOutputFeature0AsTensorBuffer().getFloatArray();

            TextView result = findViewById(R.id.view);

            result.setText("Iris-setosa: " +String.valueOf(outputFeature0[0]) + "\n" +
                    "Iris-versicolar: " + String.valueOf(outputFeature0[1]) + "\n" +
                    "Iris-virginica: " + String.valueOf(outputFeature0[2]));

            test.setText(v1+" "+v2+" "+v3+" "+" "+v4);

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exception
        }


//        try {
//            Iris model = Iris.newInstance(this);
//
//            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 4}, DataType.FLOAT32);
//            inputFeature0.loadBuffer(byteBuffer);
//
//            Iris.Outputs outputs = model.process(inputFeature0);
//            float[] outputFeature0 = outputs.getOutputFeature0AsTensorBuffer().getFloatArray();
//
//            TextView result = findViewById(R.id.view);
//
//            result.setText("Iris-setosa: " +String.valueOf(outputFeature0[0]) + "\n" +
//                    "Iris-versicolar: " + String.valueOf(outputFeature0[1]) + "\n" +
//                    "Iris-virginica: " + String.valueOf(outputFeature0[2]));
//
//            test.setText(outputFeature0[0]+" "+outputFeature0[1]+" "+outputFeature0[2]+"");
//
//            model.close();
//
//        } catch (IOException e) {
//            // TODO Handle the exception
//        }
//
    }


}