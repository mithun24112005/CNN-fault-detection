// alu.v
module alu (
    input  [31:0] a,
    input  [31:0] b,
    input  [2:0] opcode,    // 000=ADD, 001=SUB, 010=AND, 011=OR, 100=XOR
    output reg [31:0] result
);
    always @(*) begin
        case (opcode)
            3'b000: result = a + b;     // ADD
            3'b001: result = a - b;     // SUB
            3'b010: result = a & b;     // AND
            3'b011: result = a | b;     // OR
            3'b100: result = a ^ b;     // XOR
            default: result = 32'hDEAD_BEEF; // Invalid opcode
        endcase
    end
endmodule
