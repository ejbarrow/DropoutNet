plot(x2,totalvalidationerror_deep,'--*',x2,totaltrainerror_deep,'-',x1,totalvalidationerror_drop,'--*',x1,totaltrainerror_drop,'-');
    title('Graph of Validation and Train Error over time (iterations)')
    legend('Deep Validaition Error','Deep Train Error','Dropout Validaition Error','Dropout Train Error');
    xlabel('Epochs'); % x-axis label
    ylabel('Mean Squared Error'); % y-axis label