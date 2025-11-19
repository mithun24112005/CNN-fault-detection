// enhanced_alu_tb.v
`timescale 1ns/1ps
module enhanced_alu_tb;

    reg [31:0] a, b;
    reg [2:0] opcode;
    wire [31:0] result;

    reg [31:0] faulty_result;
    reg [2:0] fault_type; // 0 = no_fault, 1 = bitflip, 2 = opcode_fault
    reg [2:0] forced_opcode; // For opcode faults

    integer i;
    integer bit_to_flip;
    integer num_bitflips;
    integer j;

    // Instantiate ALU
    alu uut (
        .a(a),
        .b(b),
        .opcode(opcode),
        .result(result)
    );

    // Function to calculate correct result for given opcode
    function [31:0] calculate_correct;
        input [31:0] a_val, b_val;
        input [2:0] opcode_val;
        begin
            case (opcode_val)
                3'b000: calculate_correct = a_val + b_val;     // ADD
                3'b001: calculate_correct = a_val - b_val;     // SUB
                3'b010: calculate_correct = a_val & b_val;     // AND
                3'b011: calculate_correct = a_val | b_val;     // OR
                3'b100: calculate_correct = a_val ^ b_val;     // XOR
                default: calculate_correct = 32'hDEAD_BEEF;    // Invalid
            endcase
        end
    endfunction

    initial begin
        // Print CSV header once
        $display("a,b,opcode,faulty_result,fault_type");

        // Generate 30,000 test cases (similar to your dataset size)
        for (i = 0; i < 30000; i = i + 1) begin
            // Random inputs with better distribution
            a = $random;
            b = $random;
            opcode = $urandom_range(0, 4);  // 5 valid opcodes

            #1; // wait for ALU to compute

            // More sophisticated fault injection
            case ($urandom_range(0, 9))  // 0-9 for better distribution
                0,1,2,3: begin  // 40% No fault
                    faulty_result = result;
                    fault_type = 0;
                end
                4,5,6,7: begin  // 40% Bitflip fault (more variations)
                    num_bitflips = $urandom_range(1, 3); // 1-3 bit flips
                    faulty_result = result;
                    for (j = 0; j < num_bitflips; j = j + 1) begin
                        bit_to_flip = $urandom_range(0, 31);
                        faulty_result = faulty_result ^ (32'h1 << bit_to_flip);
                    end
                    fault_type = 1;
                end
                8,9: begin  // 20% Opcode fault (more variations)
                    forced_opcode = $urandom_range(0, 4);
                    while (forced_opcode == opcode) begin
                        forced_opcode = $urandom_range(0, 4); // Ensure different opcode
                    end
                    faulty_result = calculate_correct(a, b, forced_opcode);
                    fault_type = 2;
                end
            endcase

            // Print as CSV row to console
            $display("%0d,%0d,%0d,%0d,%0d", a, b, opcode, faulty_result, fault_type);
        end

        $finish;
    end
endmodule